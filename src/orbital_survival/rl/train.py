"""Train PPO agents on OrbitalSurvivalEnv and save each under runs/<name>_s<seed>/.

Run with `uv run --group rl python -m orbital_survival.rl.train --name foo --seed 0`;
give several seeds, as in `--seed 0 1 2`, to train them in parallel.
"""

import argparse
import json
import multiprocessing
import signal
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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
        help=f"run name; each seed is saved under {RUNS_DIR}/<name>_s<seed>/",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue each seed's saved model in the same folder; "
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
        nargs="+",
        required=True,
        help="seeds for the network and the rounds, one run each, to repeat a run",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=5,
        help="seeds trained at once; the rest wait their turn",
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
    args = parser.parse_args()
    if len(set(args.seed)) != len(args.seed):
        parser.error("--seed has a duplicate")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def run_dir_of(name: str, seed: int) -> Path:
    """Folder of one seed's run.

    Args:
        name: The --name given to train.
        seed: The seed.

    Returns:
        RUNS_DIR/<name>_s<seed>.
    """
    return RUNS_DIR / f"{name}_s{seed}"


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


_interrupted = False  # Set by Ctrl+C in a worker process


def catch_interrupt() -> None:
    """Make Ctrl+C stop training gently in this worker process.

    A KeyboardInterrupt in an idle worker would break the whole pool, so only
    set a flag, which StopOnInterrupt reads at the next step.
    """

    def handle(signum: int, frame: object) -> None:
        """Note the interrupt.

        Args:
            signum: The signal number.
            frame: The interrupted stack frame.
        """
        global _interrupted
        _interrupted = True

    signal.signal(signal.SIGINT, handle)


class StopOnInterrupt(BaseCallback):
    """Stop training at the next step after Ctrl+C."""

    def _on_step(self) -> bool:
        """Check for Ctrl+C.

        Returns:
            False to stop training.
        """
        return not _interrupted


def train_seed(args: argparse.Namespace, seed: int, quiet: bool) -> str:
    """Train and save one seed's model; run in a worker process.

    Args:
        args: The parsed arguments.
        seed: The seed to train.
        quiet: Leave out the console table, as other seeds train at once.

    Returns:
        "saved", "interrupted", or "skipped" when Ctrl+C came before it started.
    """
    tag = f"[s{seed}] "
    if _interrupted:
        return "skipped"
    run_dir = run_dir_of(args.name, seed)
    torch.set_num_threads(args.torch_threads)
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
        model = PPO.load(run_dir / MODEL_FILE, env=env, device="cpu", seed=seed)
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
            verbose=0 if quiet else 1,
            seed=seed,
            device="cpu",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_record(run_dir, args, model)
    # The console table unless quiet, plus CSV and TensorBoard copies to keep.
    console = [] if quiet else [HumanOutputFormat(sys.stdout)]
    model.set_logger(
        Logger(
            str(run_dir),
            [
                *console,
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
        verbose=2 if quiet else 0,  # A line per snapshot in place of the table
    )
    print(f"{tag}Training in {run_dir}", flush=True)
    # Without a reset the step count, and so checkpoint names, carry on.
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint, StopOnInterrupt()],
        reset_num_timesteps=not args.resume,
    )
    # Keep what was learned even when interrupted, so --resume can carry on.
    if _interrupted:
        print(f"{tag}Interrupted", flush=True)
    model.save(run_dir / MODEL_NAME)
    print(f"{tag}Saved model to {run_dir / MODEL_FILE}", flush=True)
    return "interrupted" if _interrupted else "saved"


def main() -> None:
    """Check the environment, train PPO for each seed in parallel, and save."""
    args = parse_args()
    # Check every seed first, so a bad one does not stop the others midway.
    for seed in args.seed:
        run_dir = run_dir_of(args.name, seed)
        if args.resume and not (run_dir / MODEL_FILE).exists():
            sys.exit(f"No model to resume at {run_dir / MODEL_FILE}")
        if not args.resume and run_dir.exists():
            sys.exit(f"{run_dir} already exists; add --resume or pick another --name")

    check_env(OrbitalSurvivalEnv())

    quiet = len(args.seed) > 1
    jobs = min(args.jobs, len(args.seed))
    print(f"{len(args.seed)} seeds, {jobs} at once", flush=True)
    # spawn, not fork: forking after torch has started its threads can hang.
    context = multiprocessing.get_context("spawn")
    futures: dict[int, Future[str]] = {}

    def stop(signum: int, frame: object) -> None:
        """Skip the seeds not started yet; the running ones save on their own.

        uv passes Ctrl+C on as well, so this may run twice.

        Args:
            signum: The signal number.
            frame: The interrupted stack frame.
        """
        if skipped := [seed for seed, f in futures.items() if f.cancel()]:
            print(f"Interrupted; skipping seeds {skipped}", flush=True)

    signal.signal(signal.SIGINT, stop)
    with ProcessPoolExecutor(
        jobs, mp_context=context, initializer=catch_interrupt
    ) as pool:
        for seed in args.seed:
            futures[seed] = pool.submit(train_seed, args, seed, quiet)
        wait(futures.values())

    print()
    failed = []
    for seed, future in futures.items():
        if future.cancelled():
            status = "skipped"
        elif (error := future.exception()) is not None:
            status = f"failed: {error!r}"
            failed.append(seed)
        else:
            status = future.result()
        print(f"seed {seed}: {status}  {run_dir_of(args.name, seed)}")
    if failed:
        sys.exit(f"Failed seeds: {' '.join(map(str, failed))}")


if __name__ == "__main__":
    main()
