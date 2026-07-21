#!/usr/bin/env python3
"""Generate MATH-500 rollouts on Modal (serverless GPU).

Generation only: load MATH-500 -> generate -> write rollouts_<tag>_seed<n>.jsonl
to a Modal Volume. Correctness grading runs separately and locally via
evaluate_correctness.py.

Setup (once):
  uv sync
  uv run modal setup

Run:
  uv run modal run --detach generate_rollouts.py --n 500
  uv run modal run generate_rollouts.py --n 500 --model microsoft/Phi-4-mini-instruct

Retrieve results later (from anywhere):
  modal volume get rollouts-data rollouts_reasoning_seed0.jsonl ./data/rollouts_reasoning_seed0.jsonl

Grade locally:
  uv run python evaluate_correctness.py data/rollouts_reasoning_seed0.jsonl
"""

import json
import re
import statistics
from pathlib import Path

import modal

# CUDA *devel* base ships nvcc so vLLM can JIT-compile kernels; add_python gives
# the image an interpreter (the nvidia/cuda images ship none).
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
).uv_pip_install("vllm", "transformers", "huggingface_hub", "datasets")
app = modal.App("phi-mini-rollouts", image=image)

# Persist the HF weights cache across runs so only the first cold start downloads.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# Durable output store: results survive any local disconnect.
results_vol = modal.Volume.from_name("rollouts-data", create_if_missing=True)
RESULTS_DIR = "/data"  # Volume mount inside the remote container
LOCAL_DATA_DIR = Path("data")  # where the local copy is written
MODEL = "microsoft/Phi-4-mini-reasoning"


def normalize_level(raw):
    """Coerce a MATH-500 difficulty field ("Level 3", 3, ...) to an int, or None."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else None
    return None


def model_tag(model):
    """Short slug for output filenames: reasoning / instruct / slugified basename."""
    name = model.rsplit("/", 1)[-1].lower()
    if "reasoning" in name:
        return "reasoning"
    if "instruct" in name:
        return "instruct"
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


@app.function(
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache, RESULTS_DIR: results_vol},
    timeout=2 * 60 * 60,  # a timeout loses the whole run, so be generous
)
def run_rollouts(
    n: int = 500,
    model: str = MODEL,
    max_tokens: int = 32768,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int = 0,
    output: str = None,
) -> dict:
    """Load the first `n` MATH-500 problems, generate with vLLM, and write the
    rollouts to the results Volume. Returns a generation-stats summary."""
    import os

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output = output or f"rollouts_{model_tag(model)}_seed{seed}.jsonl"

    # 1. Load MATH-500 and take the first n problems (no filtering).
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for item in ds:
        rows.append({
            "id": item.get("unique_id", str(len(rows))),
            "level": normalize_level(item.get("level")),
            "problem": item["problem"],
            "gold": item["answer"],
        })
        if len(rows) >= n:
            break
    print(f"{len(rows)} problems. Generating with {model}...")

    # 2. Generate (vLLM batches the whole set in parallel).
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=max_tokens + 4096,
    )
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,  # HF GenerationConfig defaults top_k=50; vLLM defaults -1 (off)
        max_tokens=max_tokens,
        seed=seed,
    )
    outputs = llm.generate(prompts, sp)
    hf_cache.commit()  # persist any newly cached weights

    # 3. Write rollouts to the Volume.
    n_truncated = 0
    tok_lens = []
    out_path = os.path.join(RESULTS_DIR, output)
    with open(out_path, "w") as fout:
        for r, o in zip(rows, outputs):
            gen = o.outputs[0].text
            finish_reason = o.outputs[0].finish_reason
            n_tok = len(o.outputs[0].token_ids)
            truncated = finish_reason == "length"
            n_truncated += truncated
            tok_lens.append(n_tok)
            fout.write(
                json.dumps({
                    "id": r["id"],
                    "level": r["level"],
                    "problem": r["problem"],
                    "gold": r["gold"],
                    "finish_reason": finish_reason,
                    "n_tokens": n_tok,
                    "truncated": truncated,
                    "generation": gen,
                })
                + "\n"
            )
    results_vol.commit()  # make the file durable & retrievable

    m = len(rows)
    summary = {
        "output": output,
        "model": model,
        "seed": seed,
        "n": m,
        "n_truncated": n_truncated,
        "trunc_pct": n_truncated / m,
        "max_tokens": max_tokens,
        "tokens": {
            "mean": statistics.mean(tok_lens),
            "median": statistics.median(tok_lens),
            "max": max(tok_lens),
        },
    }
    print(
        f"Truncated (hit max_tokens={max_tokens}): {n_truncated}/{m} = {n_truncated / m:.1%}"
    )
    print(
        f"Tokens/rollout: mean {summary['tokens']['mean']:.0f} "
        f"median {summary['tokens']['median']:.0f} max {summary['tokens']['max']}"
    )
    print(f"Wrote {m} rollouts -> volume 'rollouts-data':/{output}")
    return summary


@app.local_entrypoint()
def main(
    n: int = 500,
    model: str = MODEL,
    max_tokens: int = 32768,
    temperature: float = 0.6,
    seed: int = 0,
    output: str = None,
):
    """Kick off the remote generation job, print a summary, and copy the
    rollouts into the local ./data directory."""
    summary = run_rollouts.remote(
        n=n,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        output=output,
    )
    output = summary["output"]

    print(f"\nModel: {summary['model']}  (seed {summary['seed']})")
    print(
        f"Truncated: {summary['n_truncated']}/{summary['n']} "
        f"({summary['trunc_pct']:.1%}, max_tokens={summary['max_tokens']})"
    )
    print(
        f"Tokens/rollout: mean {summary['tokens']['mean']:.0f} "
        f"median {summary['tokens']['median']:.0f} max {summary['tokens']['max']}"
    )

    # Best-effort local copy; the Volume remains the source of truth if skipped.
    local_path = LOCAL_DATA_DIR / output
    try:
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in results_vol.read_file(output):
                f.write(chunk)
        print(
            f"\nSaved -> {local_path}. "
            f"Grade it next: uv run python evaluate_correctness.py {local_path}"
        )
    except Exception as e:
        print(
            f"\nResults are safe on volume 'rollouts-data':/{output} "
            f"(local copy skipped: {e}).\n"
            f"Fetch with: modal volume get rollouts-data {output} ./{local_path}"
        )
