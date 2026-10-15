"""The hostile star at the center: it spins at random and fires beams."""

import math
import random

import pygame

from orbital_survival.config import BLACK, CIRCLE_WIDTH, FPS, SUN_ORANGE, Position
from orbital_survival.entities.base import CelestialBody
from orbital_survival.entities.beam import Beam

CANNON_COUNT = 3  # Cannons spaced evenly around the star
CANNON_ARC_RANGE = math.pi * 60 / 360  # Angular width of a cannon and its beam
DECISION_INTERVAL = FPS // 8  # Frames between direction rolls and firing rolls
FIRE_CHANCE = 0.20  # Per-cannon chance to fire on a firing roll
# Weights for turning left, coasting, and turning right.
DIRECTION_WEIGHTS = [40, 20, 40]
INITIAL_SPEED_RANGE = 0.005
RECOIL_SCALE = 0.75  # Cannon shrinks to this fraction of its radius when firing
RECOIL_RECOVERY = 0.5  # Radius regained per frame afterwards


class Star(CelestialBody):
    """
    Class representing the star.
    """

    def __init__(self, center_pos: Position, size: float) -> None:
        """Set up a star with a random initial angle and spin.

        Args:
            center_pos: Center of the star.
            size: Radius of the cannons.
        """
        super().__init__(
            center_pos=center_pos,
            size=size,
            acceleration=self.ACCELERATION,
            friction=self.FRICTION,
            # Set initial angle and speed randomly
            angle=random.uniform(0, 2 * math.pi),
            speed=random.uniform(-INITIAL_SPEED_RANGE, INITIAL_SPEED_RANGE),
        )
        self.color = SUN_ORANGE
        self.arc_range = CANNON_ARC_RANGE

        # Timer and current direction for random control
        self.random_timer = 0
        self.random_direction = 0
        self.beam_timer = 0

        self.beams: list[Beam] = []
        self.cannon_initial_radius = self.size
        self.cannon_radii = [self.cannon_initial_radius] * CANNON_COUNT

    @property
    def beam_width(self) -> int:
        """Stroke width shared by the cannons and the beams they fire."""
        return int(self.size // 4)

    def update(self) -> None:
        """Advance the beams, re-roll the star's spin, and maybe fire."""
        for beam in self.beams:
            beam.update()
        self.beams = [beam for beam in self.beams if beam.is_alive()]

        self.random_timer += 1
        self.beam_timer += 1

        if self.random_timer >= DECISION_INTERVAL:
            self.random_timer = 0
            self.random_direction = random.choices(
                [-1, 0, 1], weights=DIRECTION_WEIGHTS, k=1
            )[0]

        if self.beam_timer >= DECISION_INTERVAL:
            self.beam_timer = 0
            for i in range(CANNON_COUNT):
                if random.random() < FIRE_CHANCE:
                    self.beams.append(
                        Beam(
                            self.center_pos,
                            self._cannon_angle(i),
                            self.arc_range,
                            self.size,
                            self.beam_width,
                        )
                    )
                    # Recoil: the cannon shrinks, then grows back below.
                    self.cannon_radii[i] = self.cannon_initial_radius * RECOIL_SCALE

        for i in range(CANNON_COUNT):
            if self.cannon_radii[i] < self.cannon_initial_radius:
                self.cannon_radii[i] = min(
                    self.cannon_radii[i] + RECOIL_RECOVERY, self.cannon_initial_radius
                )

        self.update_angle_and_speed(self.random_direction)

    def _cannon_angle(self, index: int) -> float:
        """Angle of the given cannon, spaced evenly around the current phase.

        Args:
            index: Cannon index, from 0 to CANNON_COUNT - 1.

        Returns:
            The cannon's angle in radians.
        """
        return self.angle + (2 * math.pi / CANNON_COUNT) * index

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the beams, then the star body, then the cannons on top.

        Args:
            screen: Surface to draw on.
        """
        for beam in self.beams:
            beam.draw(screen)

        pygame.draw.circle(screen, BLACK, self.center_pos, self.size / 2)
        pygame.draw.circle(
            screen, self.color, self.center_pos, self.size / 2, CIRCLE_WIDTH
        )

        for i in range(CANNON_COUNT):
            arc_radius = self.cannon_radii[i]
            cannon_angle = self._cannon_angle(i) % (2 * math.pi)
            start_angle = cannon_angle - self.arc_range / 2
            end_angle = cannon_angle + self.arc_range / 2
            rect = pygame.Rect(
                self.center_pos[0] - arc_radius,
                self.center_pos[1] - arc_radius,
                arc_radius * 2,
                arc_radius * 2,
            )
            pygame.draw.arc(
                screen, self.color, rect, start_angle, end_angle, self.beam_width
            )
