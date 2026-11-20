"""List the saved evaluations of a run by step, beside the random baseline.

Run with `uv run python -m orbital_survival.rl.report --name foo` after
evaluating its snapshots with `evaluate --all-steps`.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from orbital_survival.rl.paths import (
    RANDOM_RUN,
    REPORT_FILE,
    RUNS_DIR,
    evaluated_steps,
    evaluation_path,
)

CONDITIONS = ("episodes", "first_seed", "max_seconds")  # Must match to compare
SPLIT_COLUMNS = {  # List values in the summary, one column per item
    "seconds_mean_ci95": ("low", "high"),
    "seconds_quartiles": ("q1", "q2", "q3"),
}


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="List the evaluations of a run.")
    parser.add_argument(
        "--name", required=True, help="run to list, the name given to train"
    )
    return parser.parse_args()


def load_evaluations(name: str) -> dict[str, dict[str, Any]]:
    """Read the evaluations of a run and of the random policy.

    Args:
        name: Run name.

    Returns:
        Row label to evaluation record: each snapshot's steps in order, then
        "final" and "random" when they exist.
    """
    paths = {
        str(steps): evaluation_path(name, steps) for steps in evaluated_steps(name)
    }
    paths["final"] = evaluation_path(name)
    paths["random"] = evaluation_path(RANDOM_RUN)
    return {
        label: json.loads(path.read_text())
        for label, path in paths.items()
        if path.exists()
    }


def warn_mismatch(records: dict[str, dict[str, Any]]) -> None:
    """Warn when the evaluations were not played under the same conditions.

    Args:
        records: Row label to evaluation record.
    """
    for key in CONDITIONS:
        values = {label: r[key] for label, r in records.items()}
        if len(set(values.values())) > 1:
            print(f"warning: {key} differs: {values}", file=sys.stderr)


def flatten(summary: dict[str, Any]) -> dict[str, Any]:
    """Spread the list values of a summary into columns of their own.

    Args:
        summary: The "summary" of an evaluation record.

    Returns:
        Column name to value.
    """
    row = {}
    for key, value in summary.items():
        if isinstance(value, list):
            names = SPLIT_COLUMNS.get(key, tuple(map(str, range(len(value)))))
            row.update({f"{key}_{n}": v for n, v in zip(names, value, strict=True)})
        else:
            row[key] = value
    return row


def write_csv(records: dict[str, dict[str, Any]], out: Path) -> None:
    """Write one row per evaluation, with its conditions and summary.

    Args:
        records: Row label to evaluation record.
        out: The CSV file to write.
    """
    rows = [
        {"steps": label, "model_file": r["model_file"]}
        | {key: r[key] for key in CONDITIONS}
        | flatten(r["summary"])
        for label, r in records.items()
    ]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def print_table(records: dict[str, dict[str, Any]]) -> None:
    """Print one line per evaluation.

    Args:
        records: Row label to evaluation record.
    """
    first = next(iter(records.values()))
    print(
        f"{first['episodes']} rounds each (seeds {first['first_seed']} down),"
        + f" cut off at {first['max_seconds']} s"
    )
    for label, r in records.items():
        s = r["summary"]
        low, high = s["seconds_mean_ci95"]
        print(
            f"steps {label:>9}  mean {s['seconds_mean']:.1f} ({low:.1f}-{high:.1f})"
            + f"  median {s['seconds_quartiles'][1]:.1f}"
            + f"  cut off {s['cut_off']}/{r['episodes']}"
            + f"  noop {s['noop_percent']:.0f} left {s['left_percent']:.0f}"
            + f" right {s['right_percent']:.0f}  switch {s['switch_percent']:.0f}"
            + f"  entropy {s['entropy_mean']:.2f}  margin {s['margin_mean']:.2f}"
        )


def main() -> None:
    """List the evaluations of the run, print them and save them as CSV."""
    args = parse_args()
    if not (RUNS_DIR / args.name).is_dir():
        sys.exit(f"No run at {RUNS_DIR / args.name}")
    records = load_evaluations(args.name)
    if records.keys() <= {"random"}:
        sys.exit(f"No evaluations in {RUNS_DIR / args.name}")
    warn_mismatch(records)
    print_table(records)
    out = RUNS_DIR / args.name / REPORT_FILE
    write_csv(records, out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
