#!/usr/bin/env python3
"""Plot MATH-500 accuracy per model tag, averaged over seeds.

Discovers rollouts_<tag>_seed<n>_graded.jsonl files written by
evaluate_correctness.py, recomputes the per-run aggregates from their rows (the
grading summary is printed but never persisted), and renders a two-panel figure:
mean accuracy per model tag with the individual seeds overlaid, and the
truncated-rollout count behind those numbers.

Usage:
  uv run python plot_accuracy.py
  uv run python plot_accuracy.py --seeds 0 1 2 --out figures/accuracy_by_tag.png
  uv run python plot_accuracy.py rollouts data --tags reasoning instruct
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write a file
import matplotlib.pyplot as plt

GRADED_RE = re.compile(r"^rollouts_(?P<tag>.+?)_seed(?P<seed>\d+)_graded\.jsonl$")

# Tags are the subject of the comparison, so color is categorical and bound to
# the tag itself -- the same hue means the same model in both panels, and stays
# put if a tag is missing for some seed. Slots 1/2 of the reference palette.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TAG_ORDER = ["reasoning", "instruct"]  # preferred left-to-right; others append

# Published MATH-500 scores to reproduce, drawn as a reference line on the
# accuracy panel. Phi-4-Mini-Reasoning (arXiv:2504.21233) and Phi-4-Mini
# (arXiv:2503.01743).
PAPER_BASELINE = {"reasoning": 0.946, "instruct": 0.635}

# Chart chrome (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


# --- data -------------------------------------------------------------------
def discover(dirs, tags=None, seeds=None):
    """Find graded rollout files under `dirs`, keyed by (tag, seed).

    Earlier directories win, so a committed rollouts/ copy shadows a scratch
    copy in data/.
    """
    found = {}
    for d in dirs:
        for path in sorted(Path(d).glob("*_graded.jsonl")):
            m = GRADED_RE.match(path.name)
            if not m:
                continue
            key = (m["tag"], int(m["seed"]))
            if tags and key[0] not in tags:
                continue
            if seeds is not None and key[1] not in seeds:
                continue
            found.setdefault(key, path)
    return found


def summarize(path):
    """Recompute the per-run aggregates from a graded jsonl."""
    n = n_correct = n_truncated = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            n_correct += bool(row.get("correct"))
            n_truncated += bool(row.get("truncated"))
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "n_truncated": n_truncated,
    }


def collect(files):
    """Group per-run summaries by tag: {tag: [(seed, summary), ...]} sorted by seed."""
    by_tag = defaultdict(list)
    for (tag, seed), path in sorted(files.items()):
        by_tag[tag].append((seed, summarize(path)))
    for tag in by_tag:
        by_tag[tag].sort()
    ordered = [t for t in TAG_ORDER if t in by_tag]
    ordered += sorted(t for t in by_tag if t not in TAG_ORDER)
    return {t: by_tag[t] for t in ordered}


def mean(values):
    return sum(values) / len(values) if values else 0.0


# --- plotting ---------------------------------------------------------------
def style_axes(ax):
    """Recessive chrome: horizontal hairline grid, no box, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_color(INK_SECONDARY)
        label.set_fontsize(10)


def spread(count, span):
    """Evenly spaced x-offsets so overlapping seed dots stay countable."""
    if count < 2:
        return [0.0] * count
    return [-span / 2 + span * i / (count - 1) for i in range(count)]


def panel(ax, by_tag, colors, key, title, fmt, top_label_size=12, reference=None):
    """One panel: mean of `key` per tag, with every seed's run overlaid.

    Both panels share this shape, so accuracy and truncation read as the same
    kind of measurement taken twice rather than as two different charts.
    """
    tags = list(by_tag)
    runs = {tag: [s[key] for _, s in by_tag[tag]] for tag in tags}
    means = [mean(runs[tag]) for tag in tags]

    # The label offset is a constant share of the y-range, so the range has to
    # be settled before anything is drawn.
    peak = max((max(v) for v in runs.values() if v), default=0)
    # Accuracy tops out at 100% but gets a sliver of extra room so a value label
    # bumped over a reference line still has somewhere to go.
    headroom = 1.08 if key == "accuracy" else max(peak * 1.35, 1)

    ax.bar(range(len(tags)), means, width=0.38,
           color=[colors[t] for t in tags], zorder=2)

    for i, tag in enumerate(tags):
        values = runs[tag]
        top = means[i]
        # Spread across seeds as a min-max whisker; with 3 runs a stdev would
        # mostly be noise, the observed range is the honest summary.
        if len(values) > 1 and min(values) != max(values):
            ax.vlines(i, min(values), max(values), color=MUTED,
                      linewidth=1.5, zorder=3)
            top = max(top, max(values))
        # Individual seeds, ringed in the surface color so overlaps stay readable.
        ax.scatter(
            [i + dx for dx in spread(len(values), 0.14)], values, s=34,
            color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4,
        )
        # Published score as a recessive dashed rule over its own bar only, so
        # it can't be read as applying to the other tag.
        ref = (reference or {}).get(tag)
        if ref is not None:
            ax.hlines(ref, i - 0.19, i + 0.19, color=INK_SECONDARY,
                      linewidth=1.5, linestyle=(0, (4, 3)), zorder=5)
            ax.text(i + 0.24, ref, f"paper: {fmt(ref)}", ha="left", va="center",
                    color=INK_SECONDARY, fontsize=8.5)
            # Only step over the rule when it sits inside the label's own band;
            # otherwise the value would float away from the bar it belongs to.
            if top < ref < top + headroom * 0.09:
                top = ref

        ax.text(
            i, top + headroom * 0.035, fmt(means[i]), ha="center", va="bottom",
            color=INK, fontsize=top_label_size, fontweight="bold",
        )
        n_seeds = len(values)
        # x in data coords, y in axes fraction -- works whatever the y-scale is.
        ax.text(
            i, -0.075, f"{n_seeds} seed{'s' if n_seeds != 1 else ''}",
            ha="center", va="top", color=MUTED, fontsize=8.5,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(range(len(tags)), tags)
    ax.set_xlim(-0.65, len(tags) - 0.05 if reference else len(tags) - 0.35)
    ax.set_ylim(0, headroom)
    if key == "accuracy":
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold",
                 loc="left", pad=12)
    style_axes(ax)


def render(by_tag, out_path):
    colors = {tag: PALETTE[i % len(PALETTE)] for i, tag in enumerate(by_tag)}

    fig, (ax_acc, ax_trunc) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.patch.set_facecolor(SURFACE)
    # No legend: both panels name the tags on their x-axis, so identity never
    # rests on color alone.
    panel(ax_acc, by_tag, colors, "accuracy", "Mean accuracy on MATH-500",
          lambda v: f"{v:.1%}", reference=PAPER_BASELINE)
    panel(ax_trunc, by_tag, colors, "n_truncated",
          "Truncated rollouts (mean per run of 500)",
          lambda v: f"{v:.1f}", top_label_size=11)

    seeds = sorted({seed for runs in by_tag.values() for seed, _ in runs})
    fig.text(
        0.011, 0.955,
        f"Phi-4-mini reflex diffing — seeds {', '.join(map(str, seeds))}; "
        "dots are individual runs, whisker is the observed range, "
        "dashed rule is the published score",
        color=MUTED, fontsize=9, va="center",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "dirs",
        nargs="*",
        default=["rollouts", "data"],
        help="directories to search for *_graded.jsonl (default: rollouts data)",
    )
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="only these seeds (default: every seed found)")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="only these model tags (default: every tag found)")
    ap.add_argument("--out", default="figures/accuracy_by_tag.png",
                    help="output png (default: figures/accuracy_by_tag.png)")
    args = ap.parse_args()

    files = discover(args.dirs, tags=args.tags, seeds=args.seeds)
    if not files:
        raise SystemExit(
            f"No *_graded.jsonl found in {', '.join(args.dirs)}. "
            "Run evaluate_correctness.py first."
        )

    by_tag = collect(files)
    for tag, runs in by_tag.items():
        seeds = ", ".join(str(seed) for seed, _ in runs)
        accs = [s["accuracy"] for _, s in runs]
        print(f"{tag:>10}: seeds [{seeds}]  mean acc {mean(accs):.1%}")
    if args.seeds:
        missing = [
            f"{tag}/seed{seed}"
            for tag in by_tag
            for seed in args.seeds
            if (tag, seed) not in files
        ]
        if missing:
            print(f"Missing (not graded yet): {', '.join(missing)}")

    print(f"Wrote figure -> {render(by_tag, Path(args.out))}")


if __name__ == "__main__":
    main()
