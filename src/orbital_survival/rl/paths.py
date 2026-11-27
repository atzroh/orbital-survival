"""Where training runs keep their files: runs/<name>/s<seed>/ for each run."""

from pathlib import Path
from typing import Final

RUNS_DIR: Final[Path] = Path("runs")
MODEL_NAME: Final[str] = "model"  # Final model, model.zip
MODEL_FILE: Final[str] = MODEL_NAME + ".zip"
MODELS_DIR: Final[str] = "models"  # Snapshots, <steps>.zip
EVALUATIONS_DIR: Final[str] = "evaluations"  # Evaluation of each snapshot, <steps>.json
CONFIG_FILE: Final[str] = "config.json"
PROGRESS_FILE: Final[str] = "progress.csv"
EVALUATION_NAME: Final[str] = "evaluation"  # evaluation.json of the random policy
RANDOM_RUN: Final[str] = "random"  # Folder for the random policy's evaluation
SUMMARY_FILE: Final[str] = "summary.csv"  # Every condition by step, under RUNS_DIR


def run_name(name: str, seed: int) -> str:
    """Name of one seed's run, its folder relative to RUNS_DIR.

    Args:
        name: The --name given to train or evaluate.
        seed: The seed.

    Returns:
        <name>/s<seed>.
    """
    return f"{name}/s{seed}"


def model_path(name: str, steps: int | None = None) -> Path:
    """Path of a run's final model, or of one of its snapshots.

    Args:
        name: Run name, from run_name.
        steps: Step count of the snapshot to pick, or None for model.zip.

    Returns:
        The model file to load.
    """
    if steps is None:
        return RUNS_DIR / name / MODEL_FILE
    return RUNS_DIR / name / MODELS_DIR / f"{steps}.zip"


def evaluation_path(name: str, steps: int | None = None) -> Path:
    """Path of the evaluation results of a snapshot, or of the random policy.

    Args:
        name: Run name, from run_name.
        steps: Step count of the snapshot, or None for the random policy.

    Returns:
        evaluation.json, or evaluations/<steps>.json for a snapshot.
    """
    if steps is None:
        return RUNS_DIR / name / f"{EVALUATION_NAME}.json"
    return RUNS_DIR / name / EVALUATIONS_DIR / f"{steps}.json"


def snapshot_steps(name: str) -> list[int]:
    """Step counts of the snapshots a run saved, in order.

    Args:
        name: Run name, from run_name.

    Returns:
        The step counts, from the models/<steps>.zip files.
    """
    return sorted(int(f.stem) for f in (RUNS_DIR / name / MODELS_DIR).glob("*.zip"))


def evaluated_steps(name: str) -> list[int]:
    """Step counts of the snapshots a run has evaluations for, in order.

    Args:
        name: Run name, from run_name.

    Returns:
        The step counts, from the evaluations/<steps>.json files.
    """
    files = (RUNS_DIR / name / EVALUATIONS_DIR).glob("*.json")
    return sorted(int(f.stem) for f in files)
