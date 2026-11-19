"""Train a PPO agent on OrbitalSurvivalEnv and save it under runs/<name>/.

Run with `uv run --group rl python -m orbital_survival.rl.train --name foo --seed 0`.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import (
    CSVOutputFormat,
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
)

from orbital_survival.rl.env import OrbitalSurvivalEnv
from orbital_survival.rl.paths import (
    CONFIG_FILE,
    MODEL_FILE,
    MODEL_NAME,
    PROGRESS_FILE,
    RUNS_DIR,
)


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument(
        "--name",
        required=True,
        help=f"run name; everything is saved under {RUNS_DIR}/<name>/",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the run's saved model in the same folder; "
        "not recommended for --linear-lr models",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="agent steps to train (on top of the saved ones with --resume)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100_000,
        help="also save a snapshot every this many steps, in the run folder",
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[64, 64],
        help="hidden layer sizes of a new model; ignored with --resume",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="learning rate of a new model; ignored with --resume",
    )
    parser.add_argument(
        "--linear-lr",
        action="store_true",
        help="decay the learning rate linearly to 0; ignored with --resume, "
        "and a model trained with it is not meant to be resumed",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor of a new model; ignored with --resume",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="steps per env between updates of a new model; ignored with --resume",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="rounds played at once to gather experience",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="seed for the network and the rounds, to repeat a run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="minibatch size of a new model; ignored with --resume",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="passes over each batch of experience; ignored with --resume",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="entropy bonus that keeps actions random; ignored with --resume",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for PyTorch; more threads only contend with the games",
    )
    return parser.parse_args()


def start_value(value: float | Callable[[float], float]) -> float:
    """Value of a setting SB3 may hold as a schedule, at the start of training.

    Args:
        value: A number, or a schedule taking the remaining progress.

    Returns:
        The number, or the schedule at progress 1.
    """
    return value(1.0) if callable(value) else value


def ppo_settings(model: PPO) -> dict[str, Any]:
    """Hyperparameters the model actually trains with, SB3 defaults included.

    With --resume these come from the saved model, not the command line.

    Args:
        model: The model about to train.

    Returns:
        Setting name to value; schedules are given at their starting value.
    """
    clip_range_vf = model.clip_range_vf
    return {
        "learning_rate": start_value(model.learning_rate),
        "linear_lr": callable(model.learning_rate),
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "n_steps": model.n_steps,
        "n_envs": model.n_envs,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "ent_coef": model.ent_coef,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "clip_range": start_value(model.clip_range),
        "clip_range_vf": None if clip_range_vf is None else start_value(clip_range_vf),
        "normalize_advantage": model.normalize_advantage,
        "target_kl": model.target_kl,
        "net_arch": model.policy_kwargs.get("net_arch"),
        "activation_fn": model.policy.activation_fn.__name__,
        "seed": model.seed,
    }


def save_run_record(run_dir: Path, args: argparse.Namespace, model: PPO) -> None:
    """Append the conditions of this invocation to the record in the run folder.

    Args:
        run_dir: The run folder.
        args: The parsed arguments.
        model: The model about to train.
    """
    path = run_dir / CONFIG_FILE
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(
        {
            "train": {
                "timesteps": args.timesteps,
                "torch_threads": args.torch_threads,
            },
            "ppo": ppo_settings(model),
        }
    )
    path.write_text(json.dumps(history, indent=2) + "\n")


def appending_csv(path: Path) -> CSVOutputFormat:
    """Open the CSV log, carrying on the rows a resumed run already wrote.

    SB3 empties the file on open and writes columns in the order of `keys`,
    so put the old rows back and keep their column order.

    Args:
        path: The CSV file.

    Returns:
        The CSV writer for the logger.
    """
    old = path.read_text() if path.exists() else ""
    csv = CSVOutputFormat(str(path))
    if old:
        csv.file.write(old)
        csv.keys = old.split("\n", 1)[0].split(",")
    return csv


def main() -> None:
    """Check the environment, train PPO, and save the model."""
    args = parse_args()
    run_dir = RUNS_DIR / args.name
    if args.resume and not (run_dir / MODEL_FILE).exists():
        sys.exit(f"No model to resume at {run_dir / MODEL_FILE}")
    if not args.resume and run_dir.exists():
        sys.exit(f"{run_dir} already exists; add --resume or pick another --name")
    torch.set_num_threads(args.torch_threads)

    check_env(OrbitalSurvivalEnv())

    env = make_vec_env(OrbitalSurvivalEnv, n_envs=args.num_envs)
    initial_lr: float = args.learning_rate

    def linear_lr(progress: float) -> float:
        """Scale the learning rate by the remaining progress.

        Args:
            progress: Remaining progress, from 1 down to 0, given by SB3.

        Returns:
            The learning rate for the next update.
        """
        return initial_lr * progress

    learning_rate = linear_lr if args.linear_lr else initial_lr
    # A small MLP trains faster on the CPU than on a GPU.
    if args.resume:
        model = PPO.load(run_dir / MODEL_FILE, env=env, device="cpu", seed=args.seed)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            policy_kwargs={"net_arch": args.net_arch},
            verbose=1,
            seed=args.seed,
            device="cpu",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_record(run_dir, args, model)
    # The console table as before, plus CSV and TensorBoard copies to keep.
    model.set_logger(
        Logger(
            str(run_dir),
            [
                HumanOutputFormat(sys.stdout),
                appending_csv(run_dir / PROGRESS_FILE),
                TensorBoardOutputFormat(str(run_dir)),
            ],
        )
    )
    # save_freq counts calls to the vectorized env, each --num-envs steps.
    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.num_envs, 1),
        save_path=str(run_dir),
        name_prefix=MODEL_NAME,
    )
    try:
        # Without a reset the step count, and so checkpoint names, carry on.
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint,
            reset_num_timesteps=not args.resume,
        )
    except KeyboardInterrupt:
        # Keep what was learned, so --resume can carry on from here.
        print("Interrupted")
    model.save(run_dir / MODEL_NAME)
    print(f"Saved model to {run_dir / MODEL_FILE}")


if __name__ == "__main__":
    main()
