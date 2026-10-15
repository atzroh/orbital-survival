"""The scene abstraction and the modes the game can switch between."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import pygame


class GameMode(StrEnum):
    """The scenes the game can switch between."""

    START = "start"
    PLAY = "play"
    GAME_OVER = "game_over"


@dataclass(frozen=True, slots=True)
class GameResult:
    """The outcome of one finished round, shown on the game-over screen."""

    score: int
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class SceneRequest:
    """A scene's request to switch, carrying whatever the next scene needs.

    The payload rides along because the requesting scene is discarded on switch.
    """

    mode: GameMode
    result: GameResult | None = None


class Scene(ABC):
    """
    Base class for game scenes.
    """

    def __init__(self, screen: pygame.Surface) -> None:
        """Bind the scene to the surface it draws on.

        Args:
            screen: The game window's surface.
        """
        self.screen = screen

    @abstractmethod
    def handle_event(self, event: pygame.event.Event) -> SceneRequest | None:
        """Handle one event, returning a switch request, or None to stay.

        Naming a mode rather than a scene avoids circular imports between scenes.

        Args:
            event: Event to handle.

        Returns:
            A switch request, or None to stay.
        """

    @abstractmethod
    def update(self) -> SceneRequest | None:
        """Advance the scene one frame, optionally requesting a switch.

        Returns:
            A switch request, or None to stay.
        """

    @abstractmethod
    def draw(self) -> None:
        """Draw the scene onto its screen."""
