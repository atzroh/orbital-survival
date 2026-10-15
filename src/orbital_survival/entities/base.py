"""Shared bases for the objects that orbit and the arcs that are drawn."""

import math
from abc import ABC, abstractmethod

import pygame

from orbital_survival.config import (
    ACCELERATION as CFG_ACCELERATION,
    FRICTION as CFG_FRICTION,
    Color,
    Position,
)


class CelestialBody:
    """Base class for rotating celestial bodies such as planets and stars."""

    ACCELERATION: float = CFG_ACCELERATION
    FRICTION: float = CFG_FRICTION
    # Terminal angular speed: each frame adds ACCELERATION then scales by
    # FRICTION, so the speed converges on the sum of that geometric series.
    MAX_SPEED: float = ACCELERATION * FRICTION / (1 - FRICTION)

    def __init__(
        self,
        center_pos: Position,
        size: float,
        acceleration: float,
        friction: float,
        angle: float,
        speed: float,
    ) -> None:
        """Set up a body orbiting `center_pos`.

        Args:
            center_pos: Center of rotation.
            size: Radius of the body.
            acceleration: Speed added per frame of input.
            friction: Factor the speed is multiplied by each frame.
            angle: Initial angle in radians.
            speed: Initial angular speed.
        """
        self.center_pos = center_pos
        self.size = size
        self.angle = angle % (2 * math.pi)
        self.speed = speed
        self.acceleration = acceleration
        self.friction = friction

    def update_angle_and_speed(self, direction: int) -> None:
        """Accelerate in `direction`, apply friction, keep the angle in [0, 2pi).

        Args:
            direction: -1, 0 or 1; the sign of the acceleration.
        """
        self.speed += self.acceleration * direction
        self.speed *= self.friction
        self.angle += self.speed
        self.angle %= 2 * math.pi


class BaseArc(ABC):
    """Base class for arc-drawing objects (beams and beam corpses)."""

    def __init__(
        self,
        center_pos: Position,
        angle: float,
        arc_range: float,
        radius: float,
        width: int,
        color: Color,
    ) -> None:
        """Set up an arc centered on `center_pos`.

        Args:
            center_pos: Center of the arc's circle.
            angle: Angle of the arc's midpoint in radians.
            arc_range: Angular width of the arc.
            radius: Radius of the arc.
            width: Stroke width.
            color: Base color.
        """
        self.center_pos = center_pos
        self.angle = angle % (2 * math.pi)
        self.arc_range = arc_range
        self.radius = radius
        self.width = width
        self.color = color

    @abstractmethod
    def update(self) -> None:
        """Advance one frame."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return whether this object should still be simulated and drawn.

        Returns:
            True while the object is alive.
        """

    def draw_arc(self, screen: pygame.Surface, color: Color, draw_width: int) -> None:
        """Draw the arc, skipping zero-width strokes that pygame would reject.

        Args:
            screen: Surface to draw on.
            color: Color to draw with.
            draw_width: Stroke width.
        """
        if draw_width > 0:
            start_angle = self.angle - self.arc_range / 2
            end_angle = self.angle + self.arc_range / 2

            # pygame.draw.arc inscribes the arc in a bounding rectangle
            # rather than taking a center and a radius.
            rect = pygame.Rect(
                int(self.center_pos[0] - self.radius),
                int(self.center_pos[1] - self.radius),
                int(self.radius * 2),
                int(self.radius * 2),
            )
            pygame.draw.arc(screen, color, rect, start_angle, end_angle, draw_width)
