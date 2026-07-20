#!/usr/bin/env python3
"""
evaluate_correctness.py  --  local grading step for the rollouts.

Reads a rollouts_<tag>.jsonl produced by generate_rollouts.py, grades each
generation against its gold answer with math-verify, and writes an enriched
rollouts_<tag>_graded.jsonl (each row gains `correct` + `extracted_answer`).
Prints an accuracy summary.

This runs locally (no GPU, no Modal) so grading can be iterated cheaply and
independently of generation.

Usage:
  uv run python evaluate_correctness.py rollouts_reasoning.jsonl
  uv run python evaluate_correctness.py rollouts_instruct.jsonl --output graded.jsonl
"""

import argparse
import json
from pathlib import Path


# --- grading helpers (lifted verbatim from the original rollout script) ------
def extract_last_boxed(text):
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx == -1:
        return None
    i = idx + len(key)
    start, depth = i, 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def is_correct(gold, generation):
    # math_verify's parse() only extracts from delimited LaTeX/expressions; a bare
    # string (e.g. "\pi", "p - q", "\dfrac{14}{3}") parses to [] and verify()
    # then trivially fails. Wrap gold + our boxed candidate in $...$ so the LaTeX
    # and expression extractors actually fire and normalization kicks in.
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        parse,
        verify,
    )

    cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
    extracted = extract_last_boxed(generation)
    gold_p = parse(f"${gold}$", extraction_config=cfg)
    cand_p = (
        parse(f"${extracted}$", extraction_config=cfg)
        if extracted is not None
        else parse(generation, extraction_config=cfg)  # no box -> scan raw output
    )
    try:
        ok = bool(verify(gold_p, cand_p))
    except Exception:
        ok = False
    return ok, extracted


def evaluate(input_path, output_path):
    n = 0
    n_correct = 0
    n_truncated = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ok, extracted = is_correct(row["gold"], row["generation"])
            row["extracted_answer"] = extracted
            row["correct"] = ok
            n += 1
            n_correct += ok
            n_truncated += bool(row.get("truncated"))
            fout.write(json.dumps(row) + "\n")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "n_truncated": n_truncated,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        nargs="?",
        default="rollouts_reasoning.jsonl",
        help="rollouts jsonl from generate_rollouts.py",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="graded jsonl output (default: <input stem>_graded.jsonl)",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_graded.jsonl")
    )

    summary = evaluate(input_path, output_path)

    print(
        f"\nAccuracy: {summary['n_correct']}/{summary['n']} = "
        f"{summary['accuracy']:.1%}  (paper ref ~94.6%)"
    )
    print(
        f"Truncated: {summary['n_truncated']}/{summary['n']} "
        f"= {summary['n_truncated'] / summary['n']:.1%}"
        if summary["n"]
        else "No rows graded."
    )
    print(f"Wrote graded rollouts -> {summary['output']}")


if __name__ == "__main__":
    main()
