"""Train a PPO agent on OrbitalSurvivalEnv and save it.

Run with `uv run --group rl python -m orbital_survival.rl.train`.
"""

import argparse

from stable_baselines3 import PPO
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
        "--timesteps", type=int, default=500_000, help="total agent steps to train"
    )
    parser.add_argument(
        "--out",
        default="models/ppo_orbital",
        help="where to save the model (.zip is appended)",
    )
    return parser.parse_args()


def main() -> None:
    """Check the environment, train PPO, and save the model."""
    args = parse_args()

    check_env(OrbitalSurvivalEnv())

    env = make_vec_env(OrbitalSurvivalEnv, n_envs=NUM_ENVS)
    # A small MLP trains faster on the CPU than on a GPU.
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")


if __name__ == "__main__":
    main()
