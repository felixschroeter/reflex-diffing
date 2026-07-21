#!/usr/bin/env python3
"""Grade MATH-500 rollouts locally with math-verify.

Reads a rollouts_<tag>_seed<n>.jsonl produced by generate_rollouts.py, grades each
generation against its gold answer, and writes an enriched
rollouts_<tag>_seed<n>_graded.jsonl where every row gains `correct`,
`extracted_answer`, and `parse_failed`. Prints an accuracy summary. Runs locally
(no GPU, no Modal).

Usage:
  uv run python evaluate_correctness.py data/rollouts_reasoning_seed0.jsonl
  uv run python evaluate_correctness.py data/rollouts_instruct_seed0.jsonl --output data/graded.jsonl
"""

import argparse
import json
from pathlib import Path


# --- grading (math-verify) --------------------------------------------------
def extract_last_boxed(text):
    """Return the content of the last \\boxed{...} in `text`, or None."""
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
    """Grade one generation against its gold answer.

    Returns (correct, extracted_answer, parse_failed).
    """
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
    # parse_failed marks rows we couldn't reliably grade (as opposed to a genuinely
    # wrong answer): the candidate parsed to nothing, or verify() itself raised.
    parse_failed = not cand_p
    try:
        ok = bool(verify(gold_p, cand_p))
    except Exception:
        ok = False
        parse_failed = True
    return ok, extracted, parse_failed


def evaluate(input_path, output_path):
    """Grade every row of `input_path`, write the enriched rows to `output_path`,
    and return a summary dict (n, n_correct, accuracy, n_truncated, n_parse_failed)."""
    n = 0
    n_correct = 0
    n_truncated = 0
    n_parse_failed = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ok, extracted, parse_failed = is_correct(row["gold"], row["generation"])
            row["extracted_answer"] = extracted
            row["correct"] = ok
            row["parse_failed"] = parse_failed
            n += 1
            n_correct += ok
            n_truncated += bool(row.get("truncated"))
            n_parse_failed += parse_failed
            fout.write(json.dumps(row) + "\n")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "n_truncated": n_truncated,
        "n_parse_failed": n_parse_failed,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        nargs="?",
        default="data/rollouts_reasoning_seed0.jsonl",
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
    if summary["n"]:
        print(
            f"Parse failures: {summary['n_parse_failed']}/{summary['n']} "
            f"= {summary['n_parse_failed'] / summary['n']:.1%}  (grade unreliable)"
        )
        print(
            f"Truncated: {summary['n_truncated']}/{summary['n']} "
            f"= {summary['n_truncated'] / summary['n']:.1%}"
        )
    else:
        print("No rows graded.")
    print(f"Wrote graded rollouts -> {summary['output']}")


if __name__ == "__main__":
    main()
