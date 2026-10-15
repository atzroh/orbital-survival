"""Tuning constants for rendering, physics and gameplay, plus color helpers."""

import math
from typing import Final

type Color = tuple[int, int, int]
type Position = tuple[float, float]

# --- Rendering-related parameters ---

# Screen size
SCREEN_SIZE: Final[int] = 400
SCREEN_WIDTH: Final[int] = SCREEN_SIZE * 3
SCREEN_HEIGHT: Final[int] = SCREEN_SIZE * 2
# Color definitions (RGB)
WHITE: Final[Color] = (255, 255, 255)
BLACK: Final[Color] = (0, 0, 0)
GRAY: Final[Color] = (200, 200, 200)
RED: Final[Color] = (255, 0, 0)
GREEN: Final[Color] = (0, 255, 0)
EARTH_BLUE: Final[Color] = (51, 153, 204)
SUN_ORANGE: Final[Color] = (252, 130, 0)
# Frame rate
FPS: Final[int] = 120
NUM_BACKGROUND_STARS: Final[int] = 250
BUTTON_RADIUS: Final[int] = 30
# Monospaced font candidates, tried in order. A constant digit width keeps
# the HUD from jittering as values change.
FONT_NAMES: Final[list[str]] = ["consolas", "dejavusansmono", "couriernew", "monospace"]
# Scales tiny angular values into readable numbers; shared by the HUD and the
# graph so they always agree.
DISPLAY_SCALE: Final[int] = 1000

# --- Telemetry-graph parameters ---

# Same size as the game window, so the two sit side by side at the same scale.
GRAPH_WIDTH: Final[int] = SCREEN_WIDTH
GRAPH_HEIGHT: Final[int] = SCREEN_HEIGHT
# Frames of history kept and plotted. The samples are stretched across the
# plot area whatever this is, so raising it costs no screen space.
GRAPH_HISTORY_FRAMES: Final[int] = FPS * 5
# Frames between repaints; recording still happens every frame.
GRAPH_DRAW_INTERVAL: Final[int] = 4

# --- Game-system-related parameters ---

# Center of the celestial bodies, nudged up to leave room for the buttons.
CENTER_POS: Final[Position] = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
CIRCLE_WIDTH: Final[int] = 2  # Circle line width for celestial bodies

ACCELERATION: Final[float] = 0.0010  # Angular acceleration
FRICTION: Final[float] = 0.99  # Deceleration ratio

# Beam hits the player can absorb before the game ends.
MAX_LIVES: Final[int] = 3

# Beam-related parameters
BEAM_SPEED: Final[int] = 2  # Beam expansion speed
BEAM_MAX_RADIUS: Final[int] = SCREEN_SIZE  # Maximum beam radius (alive check)

# Planet-related parameters
PLANET_SIZE: Final[int] = 12  # Planet radius
PLANET_ORBIT_RADIUS: Final[int] = 225  # Planet orbital radius
PLANET_INITIAL_ANGLE: Final[float] = math.pi / 2  # 90 degrees, downward

# Star-related parameters
STAR_SIZE: Final[int] = PLANET_SIZE * 3  # Star radius

# --- Machine-learning-related parameters ---
# Max beams in state:
# (num cannons) x (BEAM_MAX_RADIUS - STAR_SIZE) / BEAM_SPEED / (fire interval)
MAX_BEAMS: Final[int] = 39


def scale_color(color: Color, ratio: float) -> Color:
    """Scale a color toward black, for fade-out effects.

    Args:
        color: Color to scale.
        ratio: Brightness factor, 0 for black and 1 for unchanged.

    Returns:
        The scaled color.
    """
    red, green, blue = color
    return int(red * ratio), int(green * ratio), int(blue * ratio)
