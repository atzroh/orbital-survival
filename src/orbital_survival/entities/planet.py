"""The player's planet: it orbits the center and trails a fading tail."""

import math

import pygame

from orbital_survival.config import (
    BLACK,
    CIRCLE_WIDTH,
    EARTH_BLUE,
    Position,
    scale_color,
)
from orbital_survival.entities.base import CelestialBody


class Planet(CelestialBody):
    """
    Class representing the planet.
    """

    # Arc swept by the trail when travelling at MAX_SPEED, and how many
    # circles are drawn along it.
    MAX_TRAJECTORY_LENGTH: float = 2 * math.pi / 6
    TRAJECTORY_NUM: int = 60

    def __init__(
        self,
        center_pos: Position,
        size: float,
        angle: float,
        orbit_radius: float,
    ) -> None:
        """Set up a planet at rest on its orbit.

        Args:
            center_pos: Center of the orbit.
            size: Radius of the planet.
            angle: Initial angle in radians.
            orbit_radius: Radius of the orbit.
        """
        super().__init__(
            center_pos=center_pos,
            size=size,
            acceleration=self.ACCELERATION,
            friction=self.FRICTION,
            angle=angle,
            speed=0.0,
        )
        self.orbit_radius = orbit_radius
        self.color = EARTH_BLUE

        # Per-frame speed delta, kept only so the HUD can display it.
        self.actual_acceleration = 0.0

        self.x, self.y = self._orbit_point(self.angle)

    def _orbit_point(self, angle: float) -> tuple[float, float]:
        """Return the point on the orbit at the given angle.

        Args:
            angle: Angle in radians.

        Returns:
            The (x, y) screen position.
        """
        return (
            self.center_pos[0] + self.orbit_radius * math.cos(angle),
            self.center_pos[1] + self.orbit_radius * math.sin(angle),
        )

    def update(self, direction: int) -> None:
        """Apply one frame of input, then move along the orbit.

        Args:
            direction: -1 for left, 0 for none, 1 for right.
        """
        speed_before_update = self.speed

        self.update_angle_and_speed(direction)

        self.actual_acceleration = self.speed - speed_before_update

        self.x, self.y = self._orbit_point(self.angle)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the trail first so the planet body sits on top of it.

        Args:
            screen: Surface to draw on.
        """
        self._draw_trajectory(screen)
        self._draw_planet(screen)

    def _draw_planet(self, screen: pygame.Surface) -> None:
        """Draw the planet as a black disc with a colored outline.

        Args:
            screen: Surface to draw on.
        """
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.size)
        pygame.draw.circle(
            screen, self.color, (int(self.x), int(self.y)), self.size, CIRCLE_WIDTH
        )

    def _draw_trajectory(self, screen: pygame.Surface) -> None:
        """Trail the planet with circles that shrink and darken behind it.

        The trail length is proportional to the current speed.

        Args:
            screen: Surface to draw on.
        """
        for n in range(self.TRAJECTORY_NUM):
            progress = n / self.TRAJECTORY_NUM
            tjy_angle = (
                self.angle
                - self.MAX_TRAJECTORY_LENGTH * (self.speed / self.MAX_SPEED) * progress
            )
            tjy_x, tjy_y = self._orbit_point(tjy_angle)
            tjy_size = self.size * (1 - progress)
            # sqrt rather than a linear ramp: perceived brightness falls off
            # faster than the numbers do, so this keeps the fade even.
            tjy_color = scale_color(self.color, math.sqrt(1 - progress))
            pygame.draw.circle(
                screen, tjy_color, (int(tjy_x), int(tjy_y)), int(tjy_size)
            )
