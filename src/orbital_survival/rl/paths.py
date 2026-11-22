"""Where training runs keep their files: one folder per run under runs/."""

from pathlib import Path
from typing import Final

RUNS_DIR: Final[Path] = Path("runs")
MODEL_NAME: Final[str] = "model"  # Snapshots are model_<steps>_steps.zip
CONFIG_FILE: Final[str] = "config.json"
PROGRESS_FILE: Final[str] = "progress.csv"
EVALUATION_NAME: Final[str] = "evaluation"  # One JSON per evaluated model file
RANDOM_RUN: Final[str] = "random"  # Folder for the random policy's evaluation
SUMMARY_FILE: Final[str] = "summary.csv"  # Every condition by step, under RUNS_DIR


def run_name(name: str, seed: int) -> str:
    """Name of one seed's run, the folder under RUNS_DIR.

    Args:
        name: The --name given to train or evaluate.
        seed: The seed.

    Returns:
        <name>_s<seed>.
    """
    return f"{name}_s{seed}"


def model_path(name: str, steps: int) -> Path:
    """Path of one of a run's snapshots.

    Args:
        name: Run name, the folder under RUNS_DIR.
        steps: Step count of the snapshot.

    Returns:
        The model file to load.
    """
    return RUNS_DIR / name / f"{MODEL_NAME}_{steps}_steps.zip"


def evaluation_path(name: str, steps: int | None = None) -> Path:
    """Path of the evaluation results of a snapshot, or of the random policy.

    Args:
        name: Run name, the folder under RUNS_DIR.
        steps: Step count of the snapshot, or None for the random policy.

    Returns:
        evaluation.json, or evaluation_<steps>_steps.json for a snapshot.
    """
    suffix = "" if steps is None else f"_{steps}_steps"
    return RUNS_DIR / name / f"{EVALUATION_NAME}{suffix}.json"


def snapshot_steps(name: str) -> list[int]:
    """Step counts of the snapshots a run saved, in order.

    Args:
        name: Run name, the folder under RUNS_DIR.

    Returns:
        The step counts, from the model_<steps>_steps.zip files.
    """
    prefix, suffix = f"{MODEL_NAME}_", "_steps.zip"
    files = (RUNS_DIR / name).glob(f"{prefix}*{suffix}")
    return sorted(int(f.name.removeprefix(prefix).removesuffix(suffix)) for f in files)


def latest_steps(name: str) -> int | None:
    """Step count of a run's newest snapshot.

    Args:
        name: Run name, the folder under RUNS_DIR.

    Returns:
        The largest step count, or None when the run has no snapshots.
    """
    return max(snapshot_steps(name), default=None)


def evaluated_steps(name: str) -> list[int]:
    """Step counts of the snapshots a run has evaluations for, in order.

    Args:
        name: Run name, the folder under RUNS_DIR.

    Returns:
        The step counts, from the evaluation_<steps>_steps.json files.
    """
    prefix, suffix = f"{EVALUATION_NAME}_", "_steps.json"
    files = (RUNS_DIR / name).glob(f"{prefix}*{suffix}")
    return sorted(int(f.name.removeprefix(prefix).removesuffix(suffix)) for f in files)
