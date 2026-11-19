"""Where training runs keep their files: one folder per run under runs/."""

from pathlib import Path
from typing import Final

RUNS_DIR: Final[Path] = Path("runs")
MODEL_NAME: Final[str] = "model"  # model.zip, plus model_<steps>_steps.zip snapshots
MODEL_FILE: Final[str] = MODEL_NAME + ".zip"
ARGS_FILE: Final[str] = "args.json"
PROGRESS_FILE: Final[str] = "progress.csv"


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
