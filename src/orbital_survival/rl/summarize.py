"""Average the evaluations of every run over its seeds, by condition and step.

A condition is the folder holding the s<seed> runs, the --name given to train.
Run with `uv run python -m orbital_survival.rl.summarize` after `evaluate`.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from orbital_survival.rl.paths import (
    RANDOM_RUN,
    RUNS_DIR,
    SUMMARY_FILE,
    evaluated_steps,
    evaluation_path,
)

SEED_DIR = re.compile(r"s\d+")
COLUMNS = [
    "condition",
    "n_seeds",
    "step",
    "seconds_mean",
    "seconds_std",
    "seconds_min",
    "seconds_max",
    "seconds_median",
    "entropy_mean",
]

type Summary = dict[str, Any]


def load_summaries() -> dict[tuple[str, int], list[Summary]]:
    """Read the summary of every evaluated snapshot.

    Returns:
        (condition, step count) to the summaries of its seeds.
    """
    groups: dict[tuple[str, int], list[Summary]] = defaultdict(list)
    for run_dir in RUNS_DIR.glob("*/s*"):
        if not run_dir.is_dir() or not SEED_DIR.fullmatch(run_dir.name):
            continue
        condition = run_dir.parent.name
        run = f"{condition}/{run_dir.name}"
        for steps in evaluated_steps(run):
            record = json.loads(evaluation_path(run, steps).read_text())
            groups[condition, steps].append(record["summary"])
    return groups


def aggregate(condition: str, steps: int, summaries: list[Summary]) -> dict[str, Any]:
    """One row of the table, from the seeds of one condition at one step.

    Args:
        condition: The condition.
        steps: The step count.
        summaries: The summary of each seed evaluated at that step.

    Returns:
        Column name to value; the spread is None with a single seed.
    """
    seconds = [s["seconds_mean"] for s in summaries]
    return {
        "condition": condition,
        "n_seeds": len(summaries),
        "step": steps,
        "seconds_mean": mean(seconds),
        "seconds_std": stdev(seconds) if len(seconds) > 1 else None,
        "seconds_min": min(seconds),
        "seconds_max": max(seconds),
        "seconds_median": mean(s["seconds_quartiles"][1] for s in summaries),
        "entropy_mean": mean(s["entropy_mean"] for s in summaries),
    }


def main() -> None:
    """Write the table of every condition and step, then the random policy."""
    if not RUNS_DIR.is_dir():
        sys.exit(f"No {RUNS_DIR} folder here")
    groups = load_summaries()
    rows = [aggregate(c, s, summaries) for (c, s), summaries in sorted(groups.items())]
    if (path := evaluation_path(RANDOM_RUN)).exists():
        s = json.loads(path.read_text())["summary"]
        rows.append(
            {
                "condition": RANDOM_RUN,
                "seconds_mean": s["seconds_mean"],
                "seconds_median": s["seconds_quartiles"][1],
                "entropy_mean": s["entropy_mean"],
            }
        )
    out = RUNS_DIR / SUMMARY_FILE
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
