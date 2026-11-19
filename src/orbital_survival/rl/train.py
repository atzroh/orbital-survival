"""Train a PPO agent on OrbitalSurvivalEnv and save it.

Run with `uv run --group rl python -m orbital_survival.rl.train --seed 0`.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from orbital_survival.rl.env import OrbitalSurvivalEnv


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="agent steps to train (on top of the saved ones with --resume)",
    )
    parser.add_argument(
        "--out",
        default="models/ppo_orbital",
        help="where to save the model (.zip is appended)",
    )
    parser.add_argument(
        "--resume",
        help="saved model to continue training from (.zip optional); "
        "not recommended for --linear-lr models",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100_000,
        help="also save a snapshot every this many steps, next to --out",
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


def save_run_history(args: argparse.Namespace, start_steps: int) -> None:
    """Append this run's arguments to the history saved next to the model.

    A resumed run carries on the history of the model it resumed from.

    Args:
        args: The parsed arguments.
        start_steps: Steps the model had already been trained for.
    """
    history = []
    if args.resume is not None:
        source = Path(args.resume.removesuffix(".zip") + ".json")
        if source.exists():
            history = json.loads(source.read_text())
    history.append(
        {
            "started": datetime.now().isoformat(timespec="seconds"),
            "start_steps": start_steps,
            "args": vars(args),
        }
    )
    path = Path(args.out + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2) + "\n")


def main() -> None:
    """Check the environment, train PPO, and save the model."""
    args = parse_args()
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
    if args.resume is None:
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
    else:
        model = PPO.load(args.resume, env=env, device="cpu", seed=args.seed)
    save_run_history(args, model.num_timesteps)
    out = Path(args.out)
    # save_freq counts calls to the vectorized env, each --num-envs steps.
    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.num_envs, 1),
        save_path=str(out.parent),
        name_prefix=out.name,
    )
    # Without a reset the step count, and so checkpoint names, carry on.
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint,
        reset_num_timesteps=args.resume is None,
    )
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")


if __name__ == "__main__":
    main()
