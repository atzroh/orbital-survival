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
# train a run named foo with seed 0, saved under runs/foo/s0 (--name and --seed are required)
uv run --group rl python -m orbital_survival.rl.train --name foo --seed 0

# train seeds 0-4 as runs/foo/s0 ... runs/foo/s4, three at a time
uv run --group rl python -m orbital_survival.rl.train --name foo --seed 0 1 2 3 4 --jobs 3

# evaluate every snapshot of runs foo/s0 to foo/s4 over fixed seeded rounds,
# several at once
uv run --group rl python -m orbital_survival.rl.evaluate --name foo --seed 0 1 2 3 4

# evaluate a random policy over the same rounds, as a baseline
uv run --group rl python -m orbital_survival.rl.evaluate --random

# average every run over its seeds, by condition and step, into runs/summary.csv
uv run python -m orbital_survival.rl.summarize

# watch the model play in the game window, with the telemetry graph beside it
uv run --group rl python -m orbital_survival.rl.watch --name foo --seed 0

# plot the training curves of every run in the browser
uv run --group rl tensorboard --logdir runs
```

`train` saves each seed under `runs/<name>/s<seed>/`, even when only one seed
is given. `evaluate` takes the same `--name` and `--seed`; `watch` takes the
same `--name` and one `--seed`.
`watch` loads the run's `model.zip`;
add `--steps 500000` to load the snapshot saved at that step count instead. Every script prints all of
its options with `--help`.

`watch` always opens the telemetry graph of the game's `--graph` (see the
top-level README) beside the game window, starting empty each round. Closing
the graph leaves the model playing. Only the window draws it, so training and
evaluation are unaffected.

## Run Folder

Each run keeps everything under `runs/<name>/s<seed>/` (ignored by git), one
folder per seed under the folder of its `--name`:

```text
runs/foo/s0
├── model.zip                # final model
├── model_100000_steps.zip   # snapshot every --checkpoint-every steps
├── config.json              # training conditions of each train invocation
├── progress.csv             # training log, one row per PPO update
├── events.out.tfevents.*    # the same log for TensorBoard
└── evaluation_<steps>_steps.json  # evaluate results of each snapshot
```

`train` refuses a name that already exists, so a run is never overwritten.
With several seeds it checks them all first and starts none if one exists.
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

`train` checks the environment, trains PPO on the CPU, saves a snapshot
every `--checkpoint-every` steps in the run folder, and saves the final model
to `model.zip` at the end. The network and PPO
hyperparameters (`--net-arch`, `--learning-rate`, `--gamma`, ...) apply to a
new model only.

Before training starts, `train` appends its conditions to `config.json`: the
step count it starts from (`from_steps`) and the steps it trains for
(`timesteps`), its PyTorch threads, and the PPO settings the model
actually trains with, SB3 defaults included. A resumed run adds an entry of its own, so the first
entry always shows how the model was created.

The training log goes to the console, and to `progress.csv` and
TensorBoard. `rollout/ep_len_mean` is the one to watch: multiply it by 8 / 120
for the seconds survived per round.

### Several Seeds

`--seed 0 1 2` trains each seed in its own process, `--jobs` of them at once
(5 by default, where the total speed levels off on a 4-core CPU); the rest
start as others finish. With two or more
seeds the console shows one line per run start, snapshot and save instead of
the table, which would interleave; `progress.csv` and TensorBoard are written
as with one seed. Each process gets `--torch-threads` PyTorch threads. A seed
that fails does not stop the others, and `train` lists every seed's outcome at
the end, exiting with an error if one failed.

Ctrl+C stops every seed at once without saving `model.zip`.

### Reproducibility

`--seed` seeds Python's `random`, NumPy, PyTorch and the first round of each
environment, so the same command repeats the same run. Training seeds in
parallel gives the same run as training each alone. The star draws from
Python's global `random`, which all environments share, so with `--num-envs` 2
or more the run is still repeatable.

### Resuming

`--name foo --seed 0 --resume` continues `runs/foo/s0/model.zip` in the same
folder, keeping its weights, optimizer state, step count and
hyperparameters. `--timesteps` is the step count the run ends at, not the
steps to add: a run saved at 5M, resumed with `--timesteps 10000000`, trains
5M more. A seed already there is left alone, so after Ctrl+C
the same command brings every seed to the same step. Snapshot names carry on the
step count, and the logs carry on too: `progress.csv` gets the new rows under
the same header, and TensorBoard draws one continuous curve.

PPO trains in whole rollouts, so a 5M-step run ends at 5,001,216 steps, and
`model.zip` keeps them all. Snapshots are still saved as the step count passes
each multiple of `--checkpoint-every` (5,100,000, ...), so their names stay on
that grid as in a run trained in one go. With `--num-envs` that does not divide
`--checkpoint-every`, a snapshot may hold a few steps more than its name.

Models trained with `--linear-lr` are not meant to be resumed. The saved
schedule is recomputed against the new total, so resuming a finished 1M-step
run up to 2M restarts the learning rate at half its initial value
instead of continuing near zero.

## Evaluation

`evaluate` plays `--episodes` rounds with every `model_<steps>_steps.zip`
snapshot of the runs `<name>/s<seed>` for each `--seed`, acting greedily. The
snapshots of all the seeds share one pool, as many at once as there are CPUs,
each in its own process with one PyTorch thread. Snapshots evaluated before
are skipped, so after `--resume` only the new ones are played; `--force` plays
them all again. It notes each snapshot as it finishes. The rounds
are seeded counting down from `2**31 - 1` (2147483647), far from the small
seeds given to `train`, so no evaluation round is one the model trained on.
Each round is cut off at fifteen minutes. `watch --round N` replays one of those
rounds again and again.

For each snapshot it works out the mean, min, max and median survival time, its
standard deviation and quartiles, a 95% confidence interval for the mean, how
many rounds reached the cut-off, the share of each action and how often the
action switches. The interval comes from a BCa bootstrap (SciPy, 10,000
resamples with a fixed seed), which suits the long right tail of survival
times better than a t interval. The shares are taken within each round and then averaged, so a long
round counts no more than a short one.

It also works out how hesitant the policy is, from the action probabilities at
each step: their entropy, which reaches ln 3 (1.10) when all three actions are
equally likely, and the margin between the two most likely actions. These are
averaged within each round and then over rounds, like the shares. The random
policy picks uniformly, so it scores ln 3 and 0. All the figures, unrounded,
go with the result of every round to `evaluation_<steps>_steps.json` in the
run folder; `--force` overwrites them.

`--random` in place of `--name` plays the same rounds with a policy that picks
each action uniformly at random (its generator seeded with 0, so it repeats),
prints every figure, and saves to `runs/random/evaluation.json` with
`model_file` set to null. Use it as the baseline a trained model should beat.

## Summary

`summarize` reads every `evaluation_<steps>_steps.json` under `runs/` without
evaluating anything, so it needs no `rl` group. It groups the runs by
condition, the folder holding the `s<seed>` runs, and writes one row per
condition and step to `runs/summary.csv`: the number of seeds evaluated at
that step, the mean, standard deviation (empty with one seed), min and max of
their `seconds_mean`, and the mean of their median survival time and of
their `entropy_mean`. The last row holds
`runs/random` when it has been evaluated.
