"""Evaluate a trained model, or a random policy, over fixed seeded rounds.

Run with `uv run --group rl python -m orbital_survival.rl.evaluate --name foo`,
or with `--random` instead of `--name` for the random policy.
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise
from statistics import mean, quantiles, stdev

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import bootstrap, entropy
from stable_baselines3 import PPO

from orbital_survival.config import FPS
from orbital_survival.rl.env import Observation, OrbitalSurvivalEnv
from orbital_survival.rl.paths import RANDOM_RUN, evaluation_path, model_path

type Probs = npt.NDArray[np.float64]  # Chance of NOOP, LEFT and RIGHT
type Policy = Callable[[Observation], tuple[np.int64, Probs]]

MAX_SECONDS = 15 * 60  # Cut off each round at fifteen minutes
FIRST_SEED = 2**31 - 1  # Counts down, far from train's --seed
BOOTSTRAP_RESAMPLES = 10_000  # Resamples for the confidence interval of the mean
UNIFORM: Probs = np.full(3, 1 / 3)  # What the random policy picks from


@dataclass(frozen=True, slots=True)
class RoundResult:
    """How one round went."""

    seed: int
    seconds: float
    cut_off: bool  # Survived until MAX_SECONDS
    actions: tuple[int, ...]  # Steps spent on NOOP, LEFT and RIGHT
    switches: int  # Steps whose action differs from the one before
    entropy: float  # Mean entropy of the action probabilities, ln 3 at most
    margin: float  # Mean gap between the two most likely actions


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--name", help="run to load, the name given to train")
    target.add_argument(
        "--random",
        action="store_true",
        help=f"play a random policy instead, saving under runs/{RANDOM_RUN}",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="load the snapshot saved at this step count instead of the final model",
    )
    parser.add_argument("--episodes", type=int, default=100, help="rounds to play")
    args = parser.parse_args()
    if args.random and args.steps is not None:
        parser.error("--steps needs --name")
    if args.episodes < 2:
        parser.error("--episodes must be at least 2 for the spread statistics")
    return args


def greedy_policy(model: PPO) -> Policy:
    """Policy that takes the model's most likely action.

    The same action as model.predict with deterministic=True.

    Args:
        model: The trained model.

    Returns:
        The policy, also giving the model's action probabilities.
    """

    def act(obs: Observation) -> tuple[np.int64, Probs]:
        """Pick the most likely action.

        Args:
            obs: The observation.

        Returns:
            The action and the probabilities it was picked from.
        """
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
        categorical = dist.distribution
        assert isinstance(categorical, torch.distributions.Categorical)
        probs = categorical.probs[0].numpy()
        return np.int64(probs.argmax()), probs.astype(np.float64)

    return act


def run_episodes(policy: Policy, episodes: int) -> list[RoundResult]:
    """Play seeded rounds counting down from FIRST_SEED with a policy.

    Args:
        policy: Maps an observation to an action and its probabilities.
        episodes: Number of rounds to play.

    Returns:
        The result of each round.
    """
    env = OrbitalSurvivalEnv(max_episode_frames=MAX_SECONDS * FPS)
    results = []
    width = len(str(episodes))
    for i, seed in enumerate(range(FIRST_SEED, FIRST_SEED - episodes, -1), 1):
        obs, info = env.reset(seed=seed)
        history: list[int] = []
        entropies: list[float] = []
        margins: list[float] = []
        done = truncated = False
        while not done:
            action, probs = policy(obs)
            history.append(int(action))
            entropies.append(float(entropy(probs)))
            second, first = np.sort(probs)[-2:]
            margins.append(float(first - second))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        result = RoundResult(
            seed=seed,
            seconds=info["frames"] / FPS,
            cut_off=truncated,
            actions=tuple(history.count(a) for a in range(3)),
            switches=sum(a != b for a, b in pairwise(history)),
            entropy=mean(entropies),
            margin=mean(margins),
        )
        results.append(result)
        cut = "  cut off" if result.cut_off else ""
        print(
            f"round {i:>{width}}/{episodes}  seed {seed}  {result.seconds:.1f} s{cut}",
            flush=True,
        )
    return results


def main() -> None:
    """Evaluate the model or the random policy, print a summary and save it."""
    args = parse_args()

    if args.random:
        run, model_file = RANDOM_RUN, None
        rng = np.random.default_rng(0)
        rounds = run_episodes(
            lambda _: (np.int64(rng.integers(3)), UNIFORM), args.episodes
        )
    else:
        path = model_path(args.name, args.steps)
        if not path.exists():
            sys.exit(f"No model at {path}")
        run, model_file = args.name, path.name
        model = PPO.load(path, device="cpu")
        # Act greedily: no exploration noise, so this measures the real skill.
        rounds = run_episodes(greedy_policy(model), args.episodes)

    sec = [r.seconds for r in rounds]
    cut_off = sum(r.cut_off for r in rounds)
    q1, q2, q3 = quantiles(sec, n=4, method="inclusive")
    # BCa corrects for the long right tail of survival times.
    ci = bootstrap(
        (sec,),
        np.mean,
        n_resamples=BOOTSTRAP_RESAMPLES,
        method="BCa",
        rng=np.random.default_rng(0),
    ).confidence_interval
    low, high = float(ci.low), float(ci.high)
    # Shares within each round, averaged so every round counts the same.
    noop, left, right = (
        mean(r.actions[i] * 100 / sum(r.actions) for r in rounds) for i in range(3)
    )
    switches = mean(r.switches * 100 / sum(r.actions) for r in rounds)
    summary = {
        "seconds_mean": mean(sec),
        "seconds_mean_ci95": [low, high],
        "seconds_std": stdev(sec),
        "seconds_min": min(sec),
        "seconds_max": max(sec),
        "seconds_quartiles": [q1, q2, q3],
        "cut_off": cut_off,
        "noop_percent": noop,
        "left_percent": left,
        "right_percent": right,
        "switch_percent": switches,
        # Means within each round, averaged over rounds like the shares.
        "entropy_mean": mean(r.entropy for r in rounds),
        "margin_mean": mean(r.margin for r in rounds),
    }

    last_seed = FIRST_SEED - args.episodes + 1
    print(f"\n{args.episodes} rounds (seeds {FIRST_SEED} down to {last_seed})")
    s = summary
    print(
        f"survival s  mean {s['seconds_mean']:.1f}"
        + f"  95% CI {low:.1f}-{high:.1f}  std {s['seconds_std']:.1f}"
        + f"  min {s['seconds_min']:.1f}  max {s['seconds_max']:.1f}"
    )
    print(f"quartiles   q1 {q1:.1f}  median {q2:.1f}  q3 {q3:.1f}")
    print(f"cut off     {s['cut_off']}/{len(rounds)}")
    print(
        f"actions %   noop {s['noop_percent']:.0f}"
        + f"  left {s['left_percent']:.0f}  right {s['right_percent']:.0f}"
    )
    print(f"switches %  {s['switch_percent']:.0f}")
    print(
        f"hesitation  entropy {s['entropy_mean']:.2f} (ln 3 = {np.log(3):.2f})"
        + f"  margin {s['margin_mean']:.2f}"
    )

    record = {
        "model_file": model_file,  # None for the random policy
        "episodes": args.episodes,
        "first_seed": FIRST_SEED,
        "max_seconds": MAX_SECONDS,
        "summary": summary,
        "rounds": [asdict(r) for r in rounds],
    }
    out = evaluation_path(run, args.steps)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
