"""Train a PPO agent on OrbitalSurvivalEnv and save it.

Run with `uv run --group rl python -m orbital_survival.rl.train`.
"""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from orbital_survival.rl.env import OrbitalSurvivalEnv

NUM_ENVS = 4  # Rounds played in parallel to gather experience


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="agent steps to train (on top of the saved ones with --resume)",
    )
    parser.add_argument(
        "--out",
        default="models/ppo_orbital",
        help="where to save the model (.zip is appended)",
    )
    parser.add_argument(
        "--resume", help="saved model to continue training from (.zip optional)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500_000,
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
        help="decay the learning rate linearly to 0 over training",
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
        "--batch-size",
        type=int,
        default=64,
        help="minibatch size of a new model; ignored with --resume",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for PyTorch; more threads only contend with the games",
    )
    return parser.parse_args()


def main() -> None:
    """Check the environment, train PPO, and save the model."""
    args = parse_args()
    torch.set_num_threads(args.torch_threads)

    check_env(OrbitalSurvivalEnv())

    env = make_vec_env(OrbitalSurvivalEnv, n_envs=NUM_ENVS)
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
            policy_kwargs={"net_arch": args.net_arch},
            verbose=1,
            device="cpu",
        )
    else:
        model = PPO.load(args.resume, env=env, device="cpu")
    out = Path(args.out)
    # save_freq counts calls to the vectorized env, each NUM_ENVS steps.
    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_every // NUM_ENVS, 1),
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
