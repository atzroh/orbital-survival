# Reinforcement Learning

A PPO agent (Stable-Baselines3) that learns to dodge beams for as long as
possible. The game logic runs headless, so training needs no window.

## Setup

SB3 and PyTorch live in the `rl` dependency group, which `uv sync` leaves out.

```bash
uv sync --group rl
```

## Run

```bash
# train (--seed is required)
uv run --group rl python -m orbital_survival.rl.train --seed 0

# compare the model with a random policy over seeded rounds
uv run --group rl python -m orbital_survival.rl.evaluate

# watch the model play in the game window
uv run --group rl python -m orbital_survival.rl.watch
```

Models are saved under `models/` (ignored by git) as `ppo_orbital.zip` by
default; pass `--out` to `train` and `--model` to `evaluate` / `watch` to use
another name. Every script prints all of its options with `--help`.

## Environment

`env.py` wraps `Logic` as a Gymnasium environment, `OrbitalSurvivalEnv`.

| Item        | Value                                                                    |
| ----------- | ------------------------------------------------------------------------ |
| Actions     | `0` coast, `1` left, `2` right                                           |
| Frame skip  | Each action is held for 8 frames (`FRAME_SKIP`)                          |
| Reward      | `+0.01` per frame survived; no penalty for a hit                         |
| Termination | The first hit ends the round, whatever `MAX_LIVES` is                    |
| Truncation  | After five minutes of game time (`MAX_EPISODE_FRAMES`)                   |
| Observation | 72 floats in `[-1, 1]`, laid out below                                   |

The observation holds only what a player can read off the screen, and every
angle is relative to the planet, so the game looks the same at any absolute
angle.

| Slice   | Features                                                                         |
| ------- | -------------------------------------------------------------------------------- |
| 1       | Planet speed                                                                     |
| 2       | Star speed, and its change over the last step (which way the star is turning)    |
| 6       | sin / cos of each of the three cannons                                           |
| 21 x 3  | Incoming beams, first to arrive first: distance to the orbit, sin, cos           |

Beam distance reads `1` as a beam leaves the star and `0` at the orbit. Unused
beam slots read as beams just leaving the star.

Changing the observation changes its shape or meaning, so models trained
before the change can no longer be loaded or used.

## Training

`train` checks the environment, trains PPO on the CPU, and saves a snapshot
every `--checkpoint-every` steps next to `--out`. The network and PPO
hyperparameters (`--net-arch`, `--learning-rate`, `--gamma`, ...) apply to a
new model only.

Each run also writes its arguments to a JSON file next to the model
(`ppo_orbital.json` by default) before training starts. A resumed run copies
the history of the model it resumed from and appends its own entry, so the
first entry always shows how the model was created.

### Reproducibility

`--seed` seeds Python's `random`, NumPy, PyTorch and the first round of each
environment, so the same command repeats the same run. The star draws from
Python's global `random`, which all environments share: with `--num-envs` 2 or
more the run is still repeatable, but environment `i` does not replay the
round that `evaluate` plays for seed `seed + i`.

### Resuming

`--resume` continues from a saved model, keeping its weights, optimizer state,
step count and hyperparameters; `--timesteps` is added on top.

Models trained with `--linear-lr` are not meant to be resumed. The saved
schedule is recomputed against the new total, so resuming a finished 1M-step
run for another 1M steps restarts the learning rate at half its initial value
instead of continuing near zero.

## Evaluation

`evaluate` plays rounds seeded `0` to `--episodes - 1` with the model (acting
greedily) and with a random policy, then prints the score and survival time
of each. It also lists the seeds where the model lost within 60 seconds;
`watch --seed N` replays one of those rounds again and again.
