#!/usr/bin/env python3
"""
generate_rollouts.py  --  Modal (serverless) version of the rollout step.

Generation ONLY: load MATH-500 -> generate -> write rollouts_<tag>.jsonl. It
runs in one remote GPU function and persists its output to a Modal Volume, so
nothing depends on your local machine staying alive -- launch detached and close
your laptop: the run survives a dropped connection or closed terminal.

Correctness grading is intentionally NOT here: it runs locally afterwards via
evaluate_correctness.py (cheap to iterate, no GPU). Reflex analysis is a later,
dedicated step and is likewise absent.

Setup (once):
  uv sync                        # local entrypoint deps (modal)
  uv run modal setup             # auth

Run (full MATH-500, detached -> survives disconnect):
  uv run modal run --detach generate_rollouts.py --n 500

  # the instruct sibling (separate output file):
  uv run modal run generate_rollouts.py --n 500 --model microsoft/Phi-4-mini-instruct

Retrieve results (any time after it finishes, even from another machine):
  modal volume get rollouts-data rollouts_reasoning.jsonl ./rollouts_reasoning.jsonl

Then grade locally:
  uv run python evaluate_correctness.py rollouts_reasoning.jsonl

Cost: a 3.8B model on an H100 is a few $ for the full 500-problem set (minutes
of GPU time), well inside the $30/mo free credits. Phi-4 is MIT-licensed -> no
HF token. Do all white-box work (activations, KL, steering) locally afterwards.
"""

import json
import re
import statistics

import modal

# --- container image --------------------------------------------------------
# CUDA *devel* base image: ships nvcc + the CUDA toolkit so vLLM can JIT-compile
# kernels / use torch.compile + CUDA graphs at runtime (the -runtime/-base tags
# omit nvcc). add_python gives the image an interpreter (the nvidia/cuda images
# ship none); pin to 3.12 for vLLM wheel coverage. Built with uv for fast,
# reproducible installs. datasets is here for loading MATH-500 remotely; grading
# deps (math-verify) now live with the local evaluate_correctness.py step.
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
).uv_pip_install(
    "vllm", "transformers", "huggingface_hub", "datasets"
)
app = modal.App("phi-mini-rollouts", image=image)

# persist the HF weights cache so cold starts after the first don't re-download
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# durable output store: results land here and survive any local disconnect
results_vol = modal.Volume.from_name("rollouts-data", create_if_missing=True)
RESULTS_DIR = "/data"
MODEL = "microsoft/Phi-4-mini-reasoning"


# --- helpers ----------------------------------------------------------------
def normalize_level(raw):
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


# --- the remote job: load -> generate -> write to Volume --------------------
@app.function(
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache, RESULTS_DIR: results_vol},
    timeout=2 * 60 * 60,  # a timeout loses the whole run, so be generous
)
def run_rollouts(
    n: int = 500,
    model: str = MODEL,
    max_tokens: int = 32768,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int = 0,
    output: str = None,
) -> dict:
    import os

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output = output or f"rollouts_{model_tag(model)}.jsonl"

    # 1. load MATH-500 (cached in hf_cache after the first run); no filtering,
    #    just take the first n problems.
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

    # 2. generate (vLLM batches the whole set in parallel)
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

    # 3. write to the Volume (durable; independent of the local session)
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
    results_vol.commit()  # <-- makes the file durable & retrievable

    m = len(rows)
    summary = {
        "output": output,
        "model": model,
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


# --- thin local entrypoint: kick off the remote job, then pull results ------
@app.local_entrypoint()
def main(
    n: int = 500,
    model: str = MODEL,
    max_tokens: int = 32768,
    temperature: float = 0.8,
    seed: int = 0,
    output: str = None,
):
    summary = run_rollouts.remote(
        n=n,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        output=output,
    )
    output = summary["output"]

    print(f"\nModel: {summary['model']}")
    print(
        f"Truncated: {summary['n_truncated']}/{summary['n']} "
        f"({summary['trunc_pct']:.1%}, max_tokens={summary['max_tokens']})"
    )
    print(
        f"Tokens/rollout: mean {summary['tokens']['mean']:.0f} "
        f"median {summary['tokens']['median']:.0f} max {summary['tokens']['max']}"
    )

    # best-effort local copy (only runs if you're still connected); the Volume
    # is the source of truth either way.
    try:
        with open(output, "wb") as f:
            for chunk in results_vol.read_file(output):
                f.write(chunk)
        print(
            f"\nSaved -> {output} (local copy). "
            f"Grade it next: uv run python evaluate_correctness.py {output}"
        )
    except Exception as e:
        print(
            f"\nResults are safe on volume 'rollouts-data':/{output} "
            f"(local copy skipped: {e}).\n"
            f"Fetch with: modal volume get rollouts-data {output} ./{output}"
        )
