"""The game-over screen: the final tally, and the way back to the title."""

import pygame

from orbital_survival.config import (
    FONT_NAMES,
    GREEN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WHITE,
)
from orbital_survival.scenes.base import GameMode, GameResult, Scene, SceneRequest
from orbital_survival.scenes.start import is_start_pressed

TITLE_FONT_SIZE = 74
RESULT_FONT_SIZE = 36
PROMPT_FONT_SIZE = 36
TITLE_OFFSET_Y = 60  # Title sits this far above the vertical center
RESULT_GAP = 40  # Gap between the title's baseline and the result line
PROMPT_GAP = 50  # Gap between the result line and the prompt
# Spaces between the two figures on the result line. A monospaced gap
# rather than a separator glyph keeps the line quiet.
RESULT_SEPARATOR = "   "


class GameOverScene(Scene):
    """
    Subclass that manages the game-over screen.
    """

    def __init__(self, screen: pygame.Surface, result: GameResult) -> None:
        """Load the fonts and keep the round's result.

        Args:
            screen: The game window's surface.
            result: Tally of the round that just ended.
        """
        super().__init__(screen)

        self.result = result
        self.title_font = pygame.font.SysFont(FONT_NAMES, TITLE_FONT_SIZE)
        self.result_font = pygame.font.SysFont(FONT_NAMES, RESULT_FONT_SIZE)
        self.prompt_font = pygame.font.SysFont(FONT_NAMES, PROMPT_FONT_SIZE)

    def handle_event(self, event: pygame.event.Event) -> SceneRequest | None:
        """Return to the title when SPACE is pressed.

        Args:
            event: Event to handle.

        Returns:
            A switch to START, or None to stay.
        """
        return SceneRequest(GameMode.START) if is_start_pressed(event) else None

    def update(self) -> SceneRequest | None:
        """The game-over screen is static; nothing to advance.

        Returns:
            Always None.
        """
        return None

    def draw(self) -> None:
        """Draw the title, the tally on one line, then the prompt."""
        title_text = self.title_font.render("GAME OVER", True, WHITE)
        title_rect = title_text.get_rect(
            center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - TITLE_OFFSET_Y)
        )
        self.screen.blit(title_text, title_rect)

        result_text = self.result_font.render(
            f"SCORE {self.result.score}"
            f"{RESULT_SEPARATOR}"
            f"TIME {self.result.elapsed_ms / 1000:.2f}s",
            True,
            WHITE,
        )
        result_rect = result_text.get_rect(
            center=(SCREEN_WIDTH / 2, title_rect.bottom + RESULT_GAP)
        )
        self.screen.blit(result_text, result_rect)

        prompt_text = self.prompt_font.render("PRESS SPACE TO CONTINUE", True, GREEN)
        prompt_rect = prompt_text.get_rect(
            center=(SCREEN_WIDTH / 2, result_rect.bottom + PROMPT_GAP)
        )
        self.screen.blit(prompt_text, prompt_rect)
