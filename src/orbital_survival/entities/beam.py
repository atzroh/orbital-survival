"""Beams fired by the star, and the red marks left where one struck."""

import pygame

from orbital_survival.config import (
    BEAM_MAX_RADIUS,
    BEAM_SPEED,
    FPS,
    PLANET_ORBIT_RADIUS,
    RED,
    WHITE,
    Position,
    scale_color,
)
from orbital_survival.entities.base import BaseArc


class Beam(BaseArc):
    """
    Class representing a beam emitted from the star.
    """

    SPEED: int = BEAM_SPEED
    MAX_RADIUS: int = BEAM_MAX_RADIUS

    def __init__(
        self,
        center_pos: Position,
        angle: float,
        arc_range: float,
        radius: float,
        width: int,
    ) -> None:
        """Set up a beam leaving the star.

        Args:
            center_pos: Center of the star.
            angle: Direction of the beam in radians.
            arc_range: Angular width of the beam.
            radius: Starting radius.
            width: Stroke width.
        """
        super().__init__(
            center_pos=center_pos,
            angle=angle,
            # Beams start at the star's surface, not at its center.
            radius=radius,
            arc_range=arc_range,
            width=width,
            color=WHITE,
        )
        self.dodged = False  # Whether this beam has already scored a dodge

    def update(self) -> None:
        """Expand the beam outward by one frame."""
        self.radius += self.SPEED

    def is_alive(self) -> bool:
        """A beam lives until it expands past the edge of the play area.

        Returns:
            True while the beam is inside the play area.
        """
        return self.radius < self.MAX_RADIUS

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the beam, fading it out over the stretch past the orbit.

        Args:
            screen: Surface to draw on.
        """
        fade_distance = self.MAX_RADIUS - PLANET_ORBIT_RADIUS

        fade_progress = (
            max(0, (self.radius - PLANET_ORBIT_RADIUS)) / fade_distance
            if fade_distance > 0
            else 1.0
        )
        life_ratio = 1.0 - min(fade_progress, 1.0)

        # Near the star the beam is thinner than its nominal width, so it
        # cannot be drawn wider than its own radius.
        draw_width = min(self.width, int(self.radius))

        self.draw_arc(screen, scale_color(self.color, life_ratio), draw_width)


class BeamCorpse(BaseArc):
    """
    Class representing a beam "corpse" shown after collision.
    """

    DURATION: int = FPS // 4  # Display duration (0.25 seconds)

    def __init__(
        self,
        center_pos: Position,
        angle: float,
        arc_range: float,
        radius: float,
        width: int,
    ) -> None:
        """Set up a corpse where a beam struck.

        Args:
            center_pos: Center of the star.
            angle: Direction of the struck beam in radians.
            arc_range: Angular width of the struck beam.
            radius: Radius at which the beam struck.
            width: Stroke width.
        """
        super().__init__(
            center_pos=center_pos,
            angle=angle,
            arc_range=arc_range,
            radius=radius,
            width=width,
            color=RED,
        )
        self.life = self.DURATION  # Frames of display time remaining

    def update(self) -> None:
        """Count down one frame of display time."""
        self.life -= 1

    def is_alive(self) -> bool:
        """Return whether the corpse still has display time left.

        Returns:
            True while display time remains.
        """
        return self.life > 0

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the corpse, dimming it as its remaining life runs out.

        Args:
            screen: Surface to draw on.
        """
        if self.is_alive():
            life_ratio = self.life / self.DURATION
            self.draw_arc(screen, scale_color(self.color, life_ratio), self.width)
