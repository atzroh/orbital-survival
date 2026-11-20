"""Evaluate a trained model over fixed seeded rounds.

Run with `uv run --group rl python -m orbital_survival.rl.evaluate --name foo`.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from itertools import pairwise
from statistics import mean, median

import numpy as np
from stable_baselines3 import PPO

from orbital_survival.config import FPS
from orbital_survival.rl.env import OrbitalSurvivalEnv
from orbital_survival.rl.paths import evaluation_path, model_path

MAX_SECONDS = 15 * 60  # Cut off each round at fifteen minutes
FIRST_SEED = 2**31 - 1  # Counts down, far from train's --seed


@dataclass(frozen=True, slots=True)
class RoundResult:
    """How one round went."""

    seed: int
    seconds: float
    cut_off: bool  # Survived until MAX_SECONDS
    actions: tuple[int, ...]  # Steps spent on NOOP, LEFT and RIGHT
    switches: int  # Steps whose action differs from the one before


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
        "--episodes", type=int, default=100, help="rounds to play per policy"
    )
    return parser.parse_args()


def run_episodes(model: PPO, episodes: int) -> list[RoundResult]:
    """Play seeded rounds counting down from FIRST_SEED with the model.

    Args:
        model: The trained model.
        episodes: Number of rounds to play.

    Returns:
        The result of each round.
    """
    env = OrbitalSurvivalEnv(max_episode_frames=MAX_SECONDS * FPS)
    results = []
    for seed in range(FIRST_SEED, FIRST_SEED - episodes, -1):
        obs, info = env.reset(seed=seed)
        history: list[int] = []
        done = truncated = False
        while not done:
            # Act greedily: no exploration noise, so this measures the real skill.
            action = np.int64(model.predict(obs, deterministic=True)[0])
            history.append(int(action))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append(
            RoundResult(
                seed=seed,
                seconds=info["frames"] / FPS,
                cut_off=truncated,
                actions=tuple(history.count(i) for i in range(3)),
                switches=sum(a != b for a, b in pairwise(history)),
            )
        )
    return results


def main() -> None:
    """Evaluate the model, print a summary and save every round."""
    args = parse_args()

    path = model_path(args.name, args.steps)
    if not path.exists():
        sys.exit(f"No model at {path}")
    model = PPO.load(path, device="cpu")
    rounds = run_episodes(model, args.episodes)

    sec = [r.seconds for r in rounds]
    cut_off = sum(r.cut_off for r in rounds)
    # Shares within each round, averaged so every round counts the same.
    noop, left, right = (
        mean(r.actions[i] * 100 / sum(r.actions) for r in rounds) for i in range(3)
    )
    switches = mean(r.switches * 100 / sum(r.actions) for r in rounds)
    summary = {
        "seconds_mean": mean(sec),
        "seconds_median": median(sec),
        "seconds_min": min(sec),
        "seconds_max": max(sec),
        "cut_off": cut_off,
        "noop_percent": noop,
        "left_percent": left,
        "right_percent": right,
        "switch_percent": switches,
    }

    last_seed = FIRST_SEED - args.episodes + 1
    print(f"{args.episodes} rounds (seeds {FIRST_SEED} down to {last_seed})")
    s = summary
    print(
        f"survival s  mean {s['seconds_mean']:.1f}"
        + f"  median {s['seconds_median']:.1f}"
        + f"  min {s['seconds_min']:.1f}  max {s['seconds_max']:.1f}"
    )
    print(f"cut off     {s['cut_off']}/{len(rounds)}")
    print(
        f"actions %   noop {s['noop_percent']:.0f}"
        + f"  left {s['left_percent']:.0f}  right {s['right_percent']:.0f}"
    )
    print(f"switches %  {s['switch_percent']:.0f}")

    record = {
        "model_file": path.name,
        "episodes": args.episodes,
        "first_seed": FIRST_SEED,
        "max_seconds": MAX_SECONDS,
        "summary": summary,
        "rounds": [asdict(r) for r in rounds],
    }
    out = evaluation_path(args.name, args.steps)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
