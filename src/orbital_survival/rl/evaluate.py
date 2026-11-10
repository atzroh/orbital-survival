"""Compare a trained model with a random policy over the same seeded rounds.

Run with `uv run --group rl python -m orbital_survival.rl.evaluate`.
"""

import argparse
from collections.abc import Callable
from statistics import mean

import numpy as np
from stable_baselines3 import PPO

from orbital_survival.config import FPS
from orbital_survival.rl.env import (
    MAX_EPISODE_FRAMES,
    Observation,
    OrbitalSurvivalEnv,
)

type Policy = Callable[[Observation], np.int64]


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument(
        "--model", default="models/ppo_orbital", help="saved model (.zip optional)"
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="rounds to play per policy"
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=MAX_EPISODE_FRAMES // FPS,
        help="cut off each round after this many seconds",
    )
    return parser.parse_args()


def run_episodes(
    policy: Policy, episodes: int, max_seconds: int
) -> list[tuple[int, float]]:
    """Play seeded rounds 0..episodes-1 with a policy.

    Args:
        policy: Maps an observation to an action.
        episodes: Number of rounds to play.
        max_seconds: Seconds after which a round is cut off.

    Returns:
        (score, seconds survived) for each round.
    """
    env = OrbitalSurvivalEnv(max_episode_frames=max_seconds * FPS)
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

    model = PPO.load(args.model, device="cpu")
    # Act greedily: no exploration noise, so this measures the real skill.
    model_results = run_episodes(
        lambda obs: np.int64(model.predict(obs, deterministic=True)[0]),
        args.episodes,
        args.max_seconds,
    )

    rng = np.random.default_rng(0)
    random_results = run_episodes(
        lambda _: np.int64(rng.integers(3)), args.episodes, args.max_seconds
    )

    print(f"{args.episodes} rounds each (seeds 0-{args.episodes - 1})")
    print("policy  score (mean/min/max)  survival s (mean/min/max)")
    print_row("model", model_results)
    print_row("random", random_results)


if __name__ == "__main__":
    main()
