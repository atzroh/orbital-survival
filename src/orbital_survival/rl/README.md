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
# train a run named foo (--name and --seed are required)
uv run --group rl python -m orbital_survival.rl.train --name foo --seed 0

# evaluate the model over fixed seeded rounds
uv run --group rl python -m orbital_survival.rl.evaluate --name foo

# evaluate a random policy over the same rounds, as a baseline
uv run --group rl python -m orbital_survival.rl.evaluate --random

# watch the model play in the game window
uv run --group rl python -m orbital_survival.rl.watch --name foo

# plot the training curves of every run in the browser
uv run --group rl tensorboard --logdir runs
```

`evaluate` and `watch` load `runs/<name>/model.zip`; add `--steps 500000` to
load the snapshot saved at that step count instead. Every script prints all of
its options with `--help`.

## Run Folder

Each run keeps everything under `runs/<name>/` (ignored by git):

```text
runs/foo
├── model.zip                # final model
├── model_100000_steps.zip   # snapshot every --checkpoint-every steps
├── config.json              # training conditions of each train invocation
├── progress.csv             # training log, one row per PPO update
├── events.out.tfevents.*    # the same log for TensorBoard
└── evaluation.json          # evaluate results (evaluation_<steps>_steps.json for a snapshot)
```

`train` refuses a name that already exists, so a run is never overwritten.
`runs/random/` holds only the `evaluation.json` of `evaluate --random`, so
`random` cannot be used as a run name once it exists.

## Environment

`env.py` wraps `Logic` as a Gymnasium environment, `OrbitalSurvivalEnv`.

| Item        | Value                                                  |
| ----------- | ------------------------------------------------------ |
| Actions     | `0` coast, `1` left, `2` right                         |
| Frame skip  | Each action is held for 8 frames (`FRAME_SKIP`)        |
| Reward      | `+0.01` per frame survived; no penalty for a hit       |
| Termination | The first hit ends the round, whatever `MAX_LIVES` is  |
| Truncation  | After five minutes of game time (`MAX_EPISODE_FRAMES`) |
| Observation | 72 floats in `[-1, 1]`, laid out below                 |

The observation holds only what a player can read off the screen, and every
angle is relative to the planet, so the game looks the same at any absolute
angle.

| Slice  | Features                                                                      |
| ------ | ----------------------------------------------------------------------------- |
| 1      | Planet speed                                                                  |
| 2      | Star speed, and its change over the last step (which way the star is turning) |
| 6      | sin / cos of each of the three cannons                                        |
| 21 x 3 | Incoming beams, first to arrive first: distance to the orbit, sin, cos        |

Beam distance reads `1` as a beam leaves the star and `0` at the orbit. Unused
beam slots read as beams just leaving the star.

Changing the observation changes its shape or meaning, so models trained
before the change can no longer be loaded or used.

## Training

`train` checks the environment, trains PPO on the CPU, and saves a snapshot
every `--checkpoint-every` steps in the run folder. The network and PPO
hyperparameters (`--net-arch`, `--learning-rate`, `--gamma`, ...) apply to a
new model only.

Before training starts, `train` appends its conditions to `config.json`: the
steps it trains for, its PyTorch threads, and the PPO settings the model
actually trains with, SB3 defaults included. A resumed run adds an entry of its own, so the first
entry always shows how the model was created.

The training log goes to the console as before, and to `progress.csv` and
TensorBoard. `rollout/ep_len_mean` is the one to watch: multiply it by 8 / 120
for the seconds survived per round.

### Reproducibility

`--seed` seeds Python's `random`, NumPy, PyTorch and the first round of each
environment, so the same command repeats the same run. The star draws from
Python's global `random`, which all environments share, so with `--num-envs` 2
or more the run is still repeatable.

### Resuming

`--name foo --resume` continues `runs/foo/model.zip` in the same folder,
keeping its weights, optimizer state, step count and hyperparameters;
`--timesteps` is added on top. Snapshot names carry on the step count, and the
logs carry on too: `progress.csv` gets the new rows under the same header, and
TensorBoard draws one continuous curve. Runs resume only from `model.zip`, not
from a snapshot.

Models trained with `--linear-lr` are not meant to be resumed. The saved
schedule is recomputed against the new total, so resuming a finished 1M-step
run for another 1M steps restarts the learning rate at half its initial value
instead of continuing near zero.

## Evaluation

`evaluate` plays `--episodes` rounds with the model, acting greedily, and
prints its survival time. The rounds
are seeded counting down from `2**31 - 1` (2147483647), far from the small
seeds given to `train`, so no evaluation round is one the model trained on.
Each round is cut off at fifteen minutes. `watch --seed N` replays one of those
rounds again and again.

Besides the mean, min and max, it prints the median survival time, how many
rounds reached the cut-off, the share of each action and how often the action
switches. The shares are taken within each round and then averaged, so a long
round counts no more than a short one. It also writes these figures, unrounded,
with the result of every round to `evaluation.json` in the run folder, or to `evaluation_<steps>_steps.json`
for a snapshot; evaluating the same model again overwrites it.

`--random` in place of `--name` plays the same rounds with a policy that picks
each action uniformly at random (its generator seeded with 0, so it repeats),
and saves to `runs/random/evaluation.json` with `model_file` set to null. Use it
as the baseline a trained model should beat.
