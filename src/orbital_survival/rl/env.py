"""Gymnasium environment that wraps the headless game logic for RL training."""

import math
import random
from typing import Any, Final, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame

from orbital_survival.config import (
    BEAM_MAX_RADIUS,
    BLACK,
    FPS,
    MAX_BEAMS,
    MAX_LIVES,
    PLANET_ORBIT_RADIUS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from orbital_survival.entities.base import CelestialBody
from orbital_survival.logic import Logic
from orbital_survival.ui.hud import HUD, HUDState

type Observation = npt.NDArray[np.float32]

# Action ids: what the agent picks each step.
NOOP: Final[int] = 0
LEFT: Final[int] = 1
RIGHT: Final[int] = 2

FRAME_SKIP: Final[int] = 4  # Frames one action is held for
SURVIVAL_REWARD: Final[float] = 0.01  # Per frame survived
HIT_PENALTY: Final[float] = 1.0  # Per life lost
MAX_EPISODE_FRAMES: Final[int] = FPS * 120  # Cut off runs at two minutes

PLANET_FEATURES: Final[int] = 4  # sin, cos, speed, lives
STAR_FEATURES: Final[int] = 3  # sin, cos, speed
BEAM_FEATURES: Final[int] = 4  # present, radial offset, sin, cos
OBS_SIZE: Final[int] = PLANET_FEATURES + STAR_FEATURES + MAX_BEAMS * BEAM_FEATURES


class OrbitalSurvivalEnv(gym.Env[Observation, np.int64]):
    """Orbital Survival as an RL environment: dodge beams for as long as possible.

    Relative angles are `planet + other`, matching `Logic._check_collisions`.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"], "render_fps": FPS}  # noqa: RUF012

    def __init__(self, render_mode: str | None = None) -> None:
        """Define the action and observation spaces; the game starts on reset.

        Args:
            render_mode: "human" to show the game in a window, or None.
        """
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.logic = Logic()
        self.frames = 0

        # Window state, created on the first frame drawn in "human" mode.
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.hud: HUD | None = None
        # Set once the window is closed, so callers can stop.
        self.quit_requested = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Start a fresh round.

        The star draws from the global `random`, so a seed reseeds it.

        Args:
            seed: Seed for reproducible rounds, or None.
            options: Unused.

        Returns:
            The first observation and the info dict.
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.logic = Logic()
        self.frames = 0
        if self.render_mode == "human":
            self.render()
        return self._observe(), self._info()

    def step(
        self, action: np.int64
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Hold the action for FRAME_SKIP frames and total up the reward.

        Args:
            action: NOOP, LEFT or RIGHT.

        Returns:
            Observation, reward, terminated, truncated and the info dict.
        """
        self.logic.left_active = action == LEFT
        self.logic.right_active = action == RIGHT

        reward = 0.0
        for _ in range(FRAME_SKIP):
            lives_before = self.logic.lives
            self.logic.update()
            self.frames += 1
            reward += SURVIVAL_REWARD
            reward -= HIT_PENALTY * (lives_before - self.logic.lives)
            # Every frame, not every step, so playback runs at game speed.
            if self.render_mode == "human":
                self.render()
            if self.logic.is_game_over:
                break

        terminated = self.logic.is_game_over
        truncated = not terminated and self.frames >= MAX_EPISODE_FRAMES
        return self._observe(), reward, terminated, truncated, self._info()

    def render(self) -> None:
        """Draw the current frame in a window, the way PlayScene does."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("ORBITAL SURVIVAL (AI)")
            self.clock = pygame.time.Clock()
            self.hud = HUD()
        assert self.clock is not None and self.hud is not None

        # Drain events so the window stays responsive and can be closed.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True

        self.screen.fill(BLACK)
        for corpse in self.logic.corpses:
            corpse.draw(self.screen)
        self.logic.star.draw(self.screen)
        self.logic.planet.draw(self.screen)
        self.hud.draw(
            self.screen,
            HUDState(
                planet_speed=self.logic.planet.speed,
                planet_acceleration=self.logic.planet.actual_acceleration,
                lives=self.logic.lives,
                score=self.logic.score,
                elapsed_ms=self.frames * 1000 // FPS,
            ),
        )
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self) -> None:
        """Close the window, if one was opened."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _observe(self) -> Observation:
        """Encode the game state as a fixed-length vector in [-1, 1].

        Returns:
            The observation.
        """
        planet = self.logic.planet
        star = self.logic.star
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        star_angle = planet.angle + star.angle
        obs[:7] = (
            math.sin(planet.angle),
            math.cos(planet.angle),
            planet.speed / CelestialBody.MAX_SPEED,
            self.logic.lives / MAX_LIVES,
            math.sin(star_angle),
            math.cos(star_angle),
            star.speed / CelestialBody.MAX_SPEED,
        )

        # Nearest to the orbit first; beams beyond MAX_BEAMS are dropped.
        beams = sorted(star.beams, key=lambda b: abs(b.radius - PLANET_ORBIT_RADIUS))
        offset = PLANET_FEATURES + STAR_FEATURES
        for i, beam in enumerate(beams[:MAX_BEAMS]):
            beam_angle = planet.angle + beam.angle
            start = offset + i * BEAM_FEATURES
            obs[start : start + BEAM_FEATURES] = (
                1.0,
                (beam.radius - PLANET_ORBIT_RADIUS) / BEAM_MAX_RADIUS,
                math.sin(beam_angle),
                math.cos(beam_angle),
            )

        return np.clip(obs, -1.0, 1.0)

    def _info(self) -> dict[str, Any]:
        """Collect values for logging; the agent never sees them.

        Returns:
            Score, lives and frames elapsed.
        """
        return {
            "score": self.logic.score,
            "lives": self.logic.lives,
            "frames": self.frames,
        }
