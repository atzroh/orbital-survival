"""A separate window plotting speed, acceleration and input over time.

Drawn on a Surface and uploaded as a texture, since a second Window has no surface.
"""

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import pygame
from pygame._sdl2.video import Renderer, Texture, Window

from orbital_survival.config import (
    ACCELERATION,
    BLACK,
    DISPLAY_SCALE,
    EARTH_BLUE,
    FONT_NAMES,
    FRICTION,
    GRAPH_DRAW_INTERVAL,
    GRAPH_HEIGHT,
    GRAPH_HISTORY_FRAMES,
    GRAPH_WIDTH,
    GRAY,
    GREEN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SUN_ORANGE,
    Color,
)
from orbital_survival.entities.base import CelestialBody

MARGIN = 10
LABEL_FONT_SIZE = 18
LABEL_INSET = 4  # Offset of the label from its pane's top-left corner

# Largest per-frame speed change: reversing at MAX_SPEED, where both the input
# and the friction terms reach ACCELERATION * FRICTION.
MAX_ACCELERATION = 2 * ACCELERATION * FRICTION


class Sample(NamedTuple):
    """One frame of the planet's motion, as the graph plots it."""

    speed: float
    acceleration: float
    direction: int


@dataclass(frozen=True, slots=True)
class Pane:
    """One of the stacked plots: how to label, scale and color a series."""

    label: str
    color: Color
    # Raw value -> displayed value, so the numbers match the HUD's.
    scale: float
    # Displayed value at the top edge of the pane; the bottom is its negative.
    value_range: float
    value_format: str


# Fixed rather than auto-scaled; the bounds follow from the tuning constants,
# so the axes stay stable and readable at a glance.
PANES = (
    Pane(
        label="SPEED",
        color=GREEN,
        scale=DISPLAY_SCALE,
        value_range=CelestialBody.MAX_SPEED * DISPLAY_SCALE,
        value_format="+08.4f",
    ),
    Pane(
        label="ACCEL",
        color=SUN_ORANGE,
        scale=DISPLAY_SCALE,
        value_range=MAX_ACCELERATION * DISPLAY_SCALE,
        value_format="+08.4f",
    ),
    Pane(
        label="INPUT",
        color=EARTH_BLUE,
        scale=1,
        value_range=1,
        value_format="+.0f",
    ),
)


class GraphRecorder:
    """The rolling history of one round, oldest sample first.

    Owned by the play scene, so each round starts empty.
    """

    def __init__(self, history_frames: int = GRAPH_HISTORY_FRAMES) -> None:
        """Start an empty history.

        Args:
            history_frames: Number of most recent frames to keep.
        """
        self.samples: deque[Sample] = deque(maxlen=history_frames)

    def record(self, speed: float, acceleration: float, direction: int) -> None:
        """Append one frame, dropping the oldest once the window is full.

        Args:
            speed: Planet speed.
            acceleration: Per-frame change in speed.
            direction: Input direction the planet received.
        """
        self.samples.append(Sample(speed, acceleration, direction))

    def series(self) -> tuple[list[float], ...]:
        """Return the three series in `PANES` order, ready to plot.

        Returns:
            Speed, acceleration and direction lists, oldest first.
        """
        return (
            [sample.speed for sample in self.samples],
            [sample.acceleration for sample in self.samples],
            [float(sample.direction) for sample in self.samples],
        )


class GraphWindow:
    """The second window, and the plot drawn into it.

    Holds the last round's recorder, so the graph stays up between rounds.
    """

    def __init__(self) -> None:
        self.window = Window(
            "ORBITAL SURVIVAL - TELEMETRY", size=(GRAPH_WIDTH, GRAPH_HEIGHT)
        )
        self.renderer = Renderer(self.window)
        self.surface = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
        self.font = pygame.font.SysFont(FONT_NAMES, LABEL_FONT_SIZE)

        self.recorder: GraphRecorder | None = None
        # Counts up to GRAPH_DRAW_INTERVAL, so the first frame paints at once
        # instead of leaving the window blank until the interval elapses.
        self.frames_since_draw = GRAPH_DRAW_INTERVAL

    def place_beside_game(self) -> None:
        """Sit beside the game window, and hand the keyboard back to it.

        Otherwise this window opens on top holding the focus, ignoring arrow keys.
        """
        game_window = Window.from_display_module()

        # Let the window manager place the new window first; a position set
        # before that is silently undone.
        pygame.event.pump()

        desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
        # Only arranged when both fit side by side; otherwise the OS placement
        # is kept so neither ends up off-screen.
        if desktop_width >= SCREEN_WIDTH + GRAPH_WIDTH:
            left = (desktop_width - SCREEN_WIDTH - GRAPH_WIDTH) // 2
            top = max(0, (desktop_height - SCREEN_HEIGHT) // 2)
            game_window.position = (left, top)
            self.window.position = (left + SCREEN_WIDTH, top)

        game_window.focus()

    def attach(self, recorder: GraphRecorder) -> None:
        """Plot this recorder from now on.

        Args:
            recorder: History of the current round.
        """
        self.recorder = recorder

    def owns(self, window: Window | None) -> bool:
        """Return whether a window event came from this window.

        Args:
            window: Window the event came from, or None.

        Returns:
            True if it is this window.
        """
        return window is self.window

    def close(self) -> None:
        """Destroy the window; the caller must stop calling `draw` afterwards."""
        self.window.destroy()

    def draw(self) -> None:
        """Repaint, but only once every GRAPH_DRAW_INTERVAL frames."""
        self.frames_since_draw += 1
        if self.frames_since_draw < GRAPH_DRAW_INTERVAL:
            return
        self.frames_since_draw = 0

        self._compose()

        texture = Texture.from_surface(self.renderer, self.surface)
        self.renderer.clear()
        texture.draw()
        self.renderer.present()

    def _compose(self) -> None:
        """Draw every pane onto the offscreen surface."""
        self.surface.fill(BLACK)

        series = self.recorder.series() if self.recorder else ((), (), ())
        for index, (pane, values) in enumerate(zip(PANES, series, strict=True)):
            self._draw_pane(pane, list(values), self._pane_rect(index))

    def _pane_rect(self, index: int) -> pygame.Rect:
        """Return the plot area of the index-th pane, counting from the top.

        Args:
            index: Pane index, from the top.

        Returns:
            The pane's plot area.
        """
        # One margin above each pane plus one below the last.
        height = (GRAPH_HEIGHT - MARGIN * (len(PANES) + 1)) // len(PANES)
        return pygame.Rect(
            MARGIN,
            MARGIN + index * (height + MARGIN),
            GRAPH_WIDTH - 2 * MARGIN,
            height,
        )

    def _draw_pane(self, pane: Pane, values: list[float], rect: pygame.Rect) -> None:
        """Draw one pane: its zero line, its trace, then its label.

        Args:
            pane: How to label, scale and color the series.
            values: Raw samples, oldest first.
            rect: Plot area of the pane.
        """
        pygame.draw.line(
            self.surface, GRAY, (rect.left, rect.centery), (rect.right, rect.centery)
        )

        # A single sample has no line to draw, and dividing by `count - 1`
        # below would fail on it.
        if len(values) > 1:
            points = [
                (self._x(index, len(values), rect), self._y(value, pane, rect))
                for index, value in enumerate(values)
            ]
            pygame.draw.lines(self.surface, pane.color, False, points)

        latest = values[-1] if values else 0.0
        label = f"{pane.label}:{latest * pane.scale:{pane.value_format}}"
        self.surface.blit(
            self.font.render(label, True, pane.color),
            (rect.left + LABEL_INSET, rect.top + LABEL_INSET),
        )

    def _x(self, index: int, count: int, rect: pygame.Rect) -> float:
        """Map a sample's position in the history onto the pane's width.

        Args:
            index: Sample position, oldest first.
            count: Number of samples; at least 2.
            rect: Plot area of the pane.

        Returns:
            Screen x coordinate.
        """
        return rect.left + index * (rect.width - 1) / (count - 1)

    def _y(self, value: float, pane: Pane, rect: pygame.Rect) -> float:
        """Map a value onto the pane's height, clamped to its bounds.

        Clamping keeps rounding error at the ACCEL bound inside the pane.

        Args:
            value: Raw sample value.
            pane: Pane giving the scale and range.
            rect: Plot area of the pane.

        Returns:
            Screen y coordinate.
        """
        offset = value * pane.scale / pane.value_range * (rect.height / 2)
        return rect.centery - max(-rect.height / 2, min(rect.height / 2, offset))
