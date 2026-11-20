"""Where training runs keep their files: one folder per run under runs/."""

from pathlib import Path
from typing import Final

RUNS_DIR: Final[Path] = Path("runs")
MODEL_NAME: Final[str] = "model"  # model.zip, plus model_<steps>_steps.zip snapshots
MODEL_FILE: Final[str] = MODEL_NAME + ".zip"
CONFIG_FILE: Final[str] = "config.json"
PROGRESS_FILE: Final[str] = "progress.csv"
EVALUATION_NAME: Final[str] = "evaluation"  # One JSON per evaluated model file
RANDOM_RUN: Final[str] = "random"  # Folder for the random policy's evaluation


def model_path(name: str, steps: int | None = None) -> Path:
    """Path of a run's final model, or of one of its snapshots.

    Args:
        name: Run name, the folder under RUNS_DIR.
        steps: Step count of the snapshot to pick, or None for model.zip.

    Returns:
        The model file to load.
    """
    file = MODEL_FILE if steps is None else f"{MODEL_NAME}_{steps}_steps.zip"
    return RUNS_DIR / name / file


def evaluation_path(name: str, steps: int | None = None) -> Path:
    """Path of the evaluation results for the model that model_path picks.

    Args:
        name: Run name, the folder under RUNS_DIR.
        steps: Step count of the snapshot, or None for the final model.

    Returns:
        evaluation.json, or evaluation_<steps>_steps.json for a snapshot.
    """
    suffix = "" if steps is None else f"_{steps}_steps"
    return RUNS_DIR / name / f"{EVALUATION_NAME}{suffix}.json"
