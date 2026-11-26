"""Watch a trained model play in the game window, round after round.

Run with `uv run --group rl python -m orbital_survival.rl.watch --name foo --seed 0`.
"""

import argparse
import sys

import numpy as np
from stable_baselines3 import PPO

from orbital_survival.config import FPS
from orbital_survival.rl.env import OrbitalSurvivalEnv
from orbital_survival.rl.paths import model_path, run_name


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play.")
    parser.add_argument("--name", required=True, help="the --name given to train")
    parser.add_argument(
        "--seed", type=int, required=True, help="seed of the run to load"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="load the snapshot saved at this step count instead of model.zip",
    )
    parser.add_argument(
        "--round",
        type=int,
        help="replay this seeded round, as in evaluate, again and again",
    )
    return parser.parse_args()


def main() -> None:
    """Play rounds until the window is closed, printing each result."""
    args = parse_args()

    run = run_name(args.name, args.seed)
    path = model_path(run, args.steps)
    if not path.exists():
        sys.exit(f"No model at {path}")
    model = PPO.load(path, device="cpu")
    env = OrbitalSurvivalEnv(render_mode="human")

    round_no = 0
    while not env.quit_requested:
        round_no += 1
        obs, info = env.reset(seed=args.round)
        done = False
        while not done and not env.quit_requested:
            action = np.int64(model.predict(obs, deterministic=True)[0])
            obs, _, terminated, _, info = env.step(action)
            done = terminated  # No cut-off: play until hit
        if done:
            seconds = info["frames"] / FPS
            print(f"round {round_no}: score {info['score']}, survived {seconds:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
