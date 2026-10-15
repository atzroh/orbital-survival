"""The play screen: wires input to the logic layer and renders the result."""

import pygame

from orbital_survival.config import BUTTON_RADIUS, SCREEN_HEIGHT, SCREEN_WIDTH
from orbital_survival.logic import Logic
from orbital_survival.scenes.base import GameMode, GameResult, Scene, SceneRequest
from orbital_survival.ui.graph import GraphRecorder, GraphWindow
from orbital_survival.ui.hud import HUD, HUDState
from orbital_survival.ui.play_button import PlayButton

BUTTON_SPACING = 100  # Horizontal offset of each button from the center
BUTTON_BOTTOM_MARGIN = 80


class PlayScene(Scene):
    """
    Subclass that manages PLAY mode.
    """

    def __init__(self, screen: pygame.Surface, graph: GraphWindow | None) -> None:
        """Start a fresh round.

        Args:
            screen: The game window's surface.
            graph: Telemetry graph to feed, or None.
        """
        super().__init__(screen)

        self.graph = graph
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Build a fresh round: logic, buttons, HUD and telemetry."""
        self.start_time = pygame.time.get_ticks()

        self.logic = Logic()

        # Circular buttons, centered on the lower edge.
        self.left_button = PlayButton(
            SCREEN_WIDTH / 2 - BUTTON_SPACING,
            SCREEN_HEIGHT - BUTTON_BOTTOM_MARGIN,
            BUTTON_RADIUS,
            "left",
        )
        self.right_button = PlayButton(
            SCREEN_WIDTH / 2 + BUTTON_SPACING,
            SCREEN_HEIGHT - BUTTON_BOTTOM_MARGIN,
            BUTTON_RADIUS,
            "right",
        )
        self.hud = HUD()

        # A round gets its own recorder, which is what makes the graph start
        # empty again without anything having to clear it.
        self.recorder = GraphRecorder()
        if self.graph is not None:
            self.graph.attach(self.recorder)

    def handle_event(self, event: pygame.event.Event) -> SceneRequest | None:
        """Play mode never switches away in response to an event.

        Args:
            event: Event to handle.

        Returns:
            Always None.
        """
        return None

    def update(self) -> SceneRequest | None:
        """Poll keyboard and mouse, then step the logic one frame.

        Input is polled so that holding a key or button keeps accelerating.

        Returns:
            A switch to GAME_OVER with the tally once the last life is spent,
            otherwise None.
        """
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()

        self.logic.left_active = keys[pygame.K_LEFT] or (
            mouse_buttons[0] and self.left_button.is_clicked(mouse_pos)
        )
        self.logic.right_active = keys[pygame.K_RIGHT] or (
            mouse_buttons[0] and self.right_button.is_clicked(mouse_pos)
        )

        self.logic.update()

        self.recorder.record(
            speed=self.logic.planet.speed,
            acceleration=self.logic.planet.actual_acceleration,
            direction=self.logic.direction,
        )

        if self.logic.is_game_over:
            return SceneRequest(
                GameMode.GAME_OVER,
                GameResult(
                    score=self.logic.score,
                    elapsed_ms=pygame.time.get_ticks() - self.start_time,
                ),
            )
        return None

    def draw(self) -> None:
        """Draw corpses, star, planet, controls and HUD, back to front."""
        for corpse in self.logic.corpses:
            corpse.draw(self.screen)

        self.logic.star.draw(self.screen)
        self.logic.planet.draw(self.screen)

        self.left_button.draw(self.screen, self.logic.left_active)
        self.right_button.draw(self.screen, self.logic.right_active)

        self.hud.draw(
            self.screen,
            HUDState(
                planet_speed=self.logic.planet.speed,
                planet_acceleration=self.logic.planet.actual_acceleration,
                lives=self.logic.lives,
                score=self.logic.score,
                elapsed_ms=pygame.time.get_ticks() - self.start_time,
            ),
        )
