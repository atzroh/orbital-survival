"""Compare a trained model with a random policy over the same seeded rounds.

Run with `uv run --group rl python -m orbital_survival.rl.evaluate --name foo`.
"""

import argparse
import sys
from collections.abc import Callable
from statistics import mean

import numpy as np
from stable_baselines3 import PPO

from orbital_survival.config import FPS
from orbital_survival.rl.env import Observation, OrbitalSurvivalEnv
from orbital_survival.rl.paths import model_path

type Policy = Callable[[Observation], np.int64]

MAX_SECONDS = 15 * 60  # Cut off each round at fifteen minutes


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument(
        "--name", required=True, help="run to load, the name given to train"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="load the snapshot saved at this step count instead of the final model",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="rounds to play per policy"
    )
    return parser.parse_args()


def run_episodes(policy: Policy, episodes: int) -> list[tuple[int, float]]:
    """Play seeded rounds 0..episodes-1 with a policy.

    Args:
        policy: Maps an observation to an action.
        episodes: Number of rounds to play.

    Returns:
        (score, seconds survived) for each round.
    """
    env = OrbitalSurvivalEnv(max_episode_frames=MAX_SECONDS * FPS)
    results = []
    for seed in range(episodes):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        results.append((info["score"], info["frames"] / FPS))
    return results


def print_row(name: str, results: list[tuple[int, float]]) -> None:
    """Print mean/min/max of score and survival time for one policy.

    Args:
        name: Label of the policy.
        results: (score, seconds) per round.
    """
    scores = [score for score, _ in results]
    seconds = [sec for _, sec in results]
    print(
        f"{name:<8}"
        f"{mean(scores):>8.0f} /{min(scores):>6} /{max(scores):>6}"
        f"{mean(seconds):>10.1f} /{min(seconds):>6.1f} /{max(seconds):>6.1f}"
    )


def main() -> None:
    """Evaluate the model and a random policy, then print a comparison."""
    args = parse_args()

    path = model_path(args.name, args.steps)
    if not path.exists():
        sys.exit(f"No model at {path}")
    model = PPO.load(path, device="cpu")
    # Act greedily: no exploration noise, so this measures the real skill.
    model_results = run_episodes(
        lambda obs: np.int64(model.predict(obs, deterministic=True)[0]),
        args.episodes,
    )

    rng = np.random.default_rng(0)
    random_results = run_episodes(lambda _: np.int64(rng.integers(3)), args.episodes)

    print(f"{args.episodes} rounds each (seeds 0-{args.episodes - 1})")
    print("policy  score (mean/min/max)  survival s (mean/min/max)")
    print_row("model", model_results)
    print_row("random", random_results)


if __name__ == "__main__":
    main()
