"""Train a PPO agent on OrbitalSurvivalEnv and save it.

Run with `uv run --group rl python -m orbital_survival.rl.train`.
"""

import argparse
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    """Check the environment, train PPO, and save the model."""
    args = parse_args()

    check_env(OrbitalSurvivalEnv())

    env = make_vec_env(OrbitalSurvivalEnv, n_envs=NUM_ENVS)
    # A small MLP trains faster on the CPU than on a GPU.
    if args.resume is None:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
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
