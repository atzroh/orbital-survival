"""Gymnasium environment that wraps the headless game logic for RL training."""

import math
import random
from typing import Any, Final, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame

from orbital_survival.config import (
    BLACK,
    FPS,
    MAX_LIVES,
    PLANET_ORBIT_RADIUS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STAR_SIZE,
)
from orbital_survival.entities import star as star_module
from orbital_survival.entities.base import CelestialBody
from orbital_survival.entities.star import CANNON_COUNT, DECISION_INTERVAL, FIRE_CHANCE
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
# A beam passing the orbit this close to the planet, but missing it, is a near miss.
NEAR_MISS_MARGIN: Final[float] = math.radians(10)
MAX_EPISODE_FRAMES: Final[int] = FPS * 120  # Cut off runs at two minutes

PLANET_FEATURES: Final[int] = 2  # speed, lives
# speed, spin direction, time to next fire roll, then sin/cos per cannon
STAR_FEATURES: Final[int] = 3 + 2 * CANNON_COUNT
BEAM_FEATURES: Final[int] = 4  # present, distance to orbit, sin, cos
# Incoming beams kept; rarely more than this are on their way at once.
BEAM_SLOTS: Final[int] = 16
OBS_SIZE: Final[int] = PLANET_FEATURES + STAR_FEATURES + BEAM_SLOTS * BEAM_FEATURES


class OrbitalSurvivalEnv(gym.Env[Observation, np.int64]):
    """Orbital Survival as an RL environment: dodge beams for as long as possible.

    Relative angles are `planet + other`, matching `Logic._check_collisions`.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"], "render_fps": FPS}  # noqa: RUF012

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_frames: int = MAX_EPISODE_FRAMES,
        near_miss_penalty: float = 0.0,
        hit_penalty: float = HIT_PENALTY,
        frame_skip: int = FRAME_SKIP,
        fire_chance: float = FIRE_CHANCE,
    ) -> None:
        """Define the action and observation spaces; the game starts on reset.

        Args:
            render_mode: "human" to show the game in a window, or None.
            max_episode_frames: Frames after which a round is cut off.
            near_miss_penalty: Penalty per frame per beam passing as a near miss.
            hit_penalty: Penalty per life lost.
            frame_skip: Frames one action is held for.
            fire_chance: Per-cannon chance to fire; above the game's makes it harder.
        """
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.max_episode_frames = max_episode_frames
        self.near_miss_penalty = near_miss_penalty
        self.hit_penalty = hit_penalty
        self.frame_skip = frame_skip
        self.fire_chance = fire_chance
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
        """Hold the action for frame_skip frames and total up the reward.

        Args:
            action: NOOP, LEFT or RIGHT.

        Returns:
            Observation, reward, terminated, truncated and the info dict.
        """
        self.logic.left_active = action == LEFT
        self.logic.right_active = action == RIGHT

        reward = 0.0
        for _ in range(self.frame_skip):
            lives_before = self.logic.lives
            # Star reads the module constant; set it per env so envs can differ.
            star_module.FIRE_CHANCE = self.fire_chance
            self.logic.update()
            self.frames += 1
            reward += SURVIVAL_REWARD
            reward -= self.hit_penalty * (lives_before - self.logic.lives)
            if self.near_miss_penalty:
                reward -= self.near_miss_penalty * self._near_misses()
            # Every frame, not every step, so playback runs at game speed.
            if self.render_mode == "human":
                self.render()
            if self.logic.is_game_over:
                break

        terminated = self.logic.is_game_over
        truncated = not terminated and self.frames >= self.max_episode_frames
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

        # Everything is relative to the planet; the game looks the same at
        # any absolute angle.
        cannons = []
        for i in range(CANNON_COUNT):
            cannon_angle = planet.angle + star.angle + 2 * math.pi * i / CANNON_COUNT
            cannons += [math.sin(cannon_angle), math.cos(cannon_angle)]
        obs[: PLANET_FEATURES + STAR_FEATURES] = (
            planet.speed / CelestialBody.MAX_SPEED,
            self.logic.lives / MAX_LIVES,
            star.speed / CelestialBody.MAX_SPEED,
            star.random_direction,
            (DECISION_INTERVAL - star.beam_timer) / DECISION_INTERVAL,
            *cannons,
        )

        # Only beams that can still hit, the first to arrive first. The
        # back-edge test matches the band in Logic._check_collisions.
        incoming = [
            beam
            for beam in star.beams
            if beam.radius - beam.width - planet.size < PLANET_ORBIT_RADIUS
        ]
        incoming.sort(key=lambda b: b.radius, reverse=True)
        offset = PLANET_FEATURES + STAR_FEATURES
        for i, beam in enumerate(incoming[:BEAM_SLOTS]):
            beam_angle = planet.angle + beam.angle
            start = offset + i * BEAM_FEATURES
            obs[start : start + BEAM_FEATURES] = (
                1.0,
                # 1 as it leaves the star, 0 at the orbit, slightly negative
                # while it crosses.
                (PLANET_ORBIT_RADIUS - beam.radius) / (PLANET_ORBIT_RADIUS - STAR_SIZE),
                math.sin(beam_angle),
                math.cos(beam_angle),
            )

        return np.clip(obs, -1.0, 1.0)

    def _near_misses(self) -> int:
        """Count beams crossing the orbit that only just miss the planet.

        Uses the same band and angles as `Logic._check_collisions`.

        Returns:
            The number of near misses this frame.
        """
        planet = self.logic.planet
        margin = math.asin(planet.size / planet.orbit_radius)
        count = 0
        for beam in self.logic.star.beams:
            back = beam.radius - beam.width - planet.size
            front = beam.radius + beam.width + planet.size
            if not back < planet.orbit_radius < front:
                continue
            diff = (planet.angle + beam.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < beam.arc_range / 2 + margin + NEAR_MISS_MARGIN:
                count += 1
        return count

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
