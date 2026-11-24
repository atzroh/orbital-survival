"""The static starfield behind the game, shared by the game and the RL window."""

import random
from typing import Final, NamedTuple

import pygame

from orbital_survival.config import SCREEN_HEIGHT, SCREEN_WIDTH, Color, Position

# Mostly 1px stars with the occasional 2px one, at a dim brightness range
# that keeps them from competing with the gameplay.
STAR_RADII: Final[list[int]] = [1, 1, 1, 2]
STAR_BRIGHTNESS_RANGE: Final[tuple[int, int]] = (50, 150)
# A private stream, so the starfield never shifts the game's global `random`.
STARFIELD_SEED: Final[int] = 0


class BackgroundStar(NamedTuple):
    """One static speck of the starfield behind the game."""

    pos: Position
    radius: int
    color: Color


def create_stars(num_stars: int) -> list[BackgroundStar]:
    """Scatter the starfield, the same every time; it never moves afterwards.

    Args:
        num_stars: Number of stars to place.

    Returns:
        The background stars.
    """
    rng = random.Random(STARFIELD_SEED)
    stars = []
    for _ in range(num_stars):
        x = rng.randint(0, SCREEN_WIDTH)
        y = rng.randint(0, SCREEN_HEIGHT)
        radius = rng.choice(STAR_RADII)
        brightness = rng.randint(*STAR_BRIGHTNESS_RANGE)
        stars.append(
            BackgroundStar(
                pos=(x, y),
                radius=radius,
                color=(brightness, brightness, brightness),
            )
        )
    return stars


def draw_stars(screen: pygame.Surface, stars: list[BackgroundStar]) -> None:
    """Draw the starfield onto the screen.

    Args:
        screen: Surface to draw on.
        stars: The background stars.
    """
    for star in stars:
        pygame.draw.circle(screen, star.color, star.pos, star.radius)
