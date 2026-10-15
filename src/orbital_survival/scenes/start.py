"""The title screen, which waits for SPACE and then hands off to play."""

import pygame

from orbital_survival.config import (
    FONT_NAMES,
    GREEN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WHITE,
)
from orbital_survival.scenes.base import GameMode, Scene, SceneRequest

TITLE_FONT_SIZE = 74
PROMPT_FONT_SIZE = 36
TITLE_OFFSET_Y = 50  # Title sits this far above the vertical center
PROMPT_GAP = 30  # Gap between the title's baseline and the prompt


def is_start_pressed(event: pygame.event.Event) -> bool:
    """Return True when the event is a SPACE key press.

    Args:
        event: Event to inspect.

    Returns:
        True for a SPACE key press.
    """
    return event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE


class StartScene(Scene):
    """
    Subclass that manages the start screen.
    """

    def __init__(self, screen: pygame.Surface) -> None:
        """Load the title screen's fonts.

        Args:
            screen: The game window's surface.
        """
        super().__init__(screen)

        self.title_font = pygame.font.SysFont(FONT_NAMES, TITLE_FONT_SIZE)
        self.prompt_font = pygame.font.SysFont(FONT_NAMES, PROMPT_FONT_SIZE)

    def handle_event(self, event: pygame.event.Event) -> SceneRequest | None:
        """Switch to PLAY when SPACE is pressed.

        Args:
            event: Event to handle.

        Returns:
            A switch to PLAY, or None to stay.
        """
        return SceneRequest(GameMode.PLAY) if is_start_pressed(event) else None

    def update(self) -> SceneRequest | None:
        """The title screen is static; nothing to advance.

        Returns:
            Always None.
        """
        return None

    def draw(self) -> None:
        """Draw the title, with the prompt centered beneath it."""
        title_text = self.title_font.render("ORBITAL SURVIVAL", True, WHITE)
        title_rect = title_text.get_rect(
            center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - TITLE_OFFSET_Y)
        )
        self.screen.blit(title_text, title_rect)

        prompt_text = self.prompt_font.render("PRESS SPACE TO PLAY", True, GREEN)
        prompt_rect = prompt_text.get_rect(
            center=(SCREEN_WIDTH / 2, title_rect.bottom + PROMPT_GAP)
        )
        self.screen.blit(prompt_text, prompt_rect)
