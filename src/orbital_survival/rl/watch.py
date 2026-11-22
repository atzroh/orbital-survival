"""Watch a trained model play in the game window, round after round.

Run with `uv run --group rl python -m orbital_survival.rl.watch --name foo`.
"""

import argparse
import sys

import numpy as np
from stable_baselines3 import PPO

from orbital_survival.config import FPS
from orbital_survival.rl.env import OrbitalSurvivalEnv
from orbital_survival.rl.paths import RUNS_DIR, latest_steps, model_path


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play.")
    parser.add_argument(
        "--name", required=True, help="run to load, the name given to train"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="load the snapshot saved at this step count instead of the newest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="replay this seeded round, as in evaluate, again and again",
    )
    return parser.parse_args()


def main() -> None:
    """Play rounds until the window is closed, printing each result."""
    args = parse_args()

    steps = args.steps if args.steps is not None else latest_steps(args.name)
    if steps is None:
        sys.exit(f"No snapshots in {RUNS_DIR / args.name}")
    path = model_path(args.name, steps)
    if not path.exists():
        sys.exit(f"No model at {path}")
    model = PPO.load(path, device="cpu")
    env = OrbitalSurvivalEnv(render_mode="human")

    round_no = 0
    while not env.quit_requested:
        round_no += 1
        obs, info = env.reset(seed=args.seed)
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
