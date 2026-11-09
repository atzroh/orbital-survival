# Backlog

Follow-ups for the reinforcement-learning work (`src/orbital_survival/rl/`).

## Evaluation and visualization

- [ ] Evaluation script: run a saved model for N episodes and report the mean
      score and survival time, side by side with a random policy.
- [ ] Viewer mode: let a saved model play in the pygame window
      (`render_mode="human"`), so its behavior can be watched.

## Training

- [ ] Reproducibility: pass a `random.Random` into `Logic` / `Star` instead of
      reseeding the global `random`, so each environment has its own stream.
- [ ] Parallelism: try `SubprocVecEnv` to use more CPU cores.
- [ ] TensorBoard logging, to plot `ep_len_mean` and `ep_rew_mean` over time.
- [ ] Hyperparameter tuning (learning rate, `n_steps`, network size).
- [ ] Compare reward designs (e.g. game score, distance-to-beam shaping) and
      `FRAME_SKIP` values.
- [ ] Compare PPO with DQN.
- [ ] Ablate the v2 observation changes (incoming-only beams, all three
      cannons, no absolute angle, fire timer and spin direction) to see which
      ones actually help.

## Learning

- [ ] Implement DQN / PPO from scratch in PyTorch and compare with SB3.

## Maintenance

- [ ] Tests for the environment: observation shape and range, reward on a hit.
- [ ] README: document how to install the `rl` group and run training.
