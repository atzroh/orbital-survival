"""Watch a trained model play in the game window, round after round.

Run with `uv run --group rl python -m orbital_survival.rl.watch`.
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

from orbital_survival.config import FPS
from orbital_survival.rl.env import OrbitalSurvivalEnv


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play.")
    parser.add_argument(
        "--model", default="models/ppo_orbital", help="saved model (.zip optional)"
    )
    return parser.parse_args()


def main() -> None:
    """Play rounds until the window is closed, printing each result."""
    args = parse_args()

    model = PPO.load(args.model, device="cpu")
    env = OrbitalSurvivalEnv(render_mode="human")

    round_no = 0
    while not env.quit_requested:
        round_no += 1
        obs, info = env.reset()
        done = False
        while not done and not env.quit_requested:
            action = np.int64(model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if done:
            seconds = info["frames"] / FPS
            print(f"round {round_no}: score {info['score']}, survived {seconds:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
