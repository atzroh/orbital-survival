"""Window setup, the main loop, and switching between scenes."""

import pygame

from orbital_survival.config import (
    BLACK,
    FPS,
    NUM_BACKGROUND_STARS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from orbital_survival.scenes.base import GameMode, Scene, SceneRequest
from orbital_survival.scenes.game_over import GameOverScene
from orbital_survival.scenes.play import PlayScene
from orbital_survival.scenes.start import StartScene
from orbital_survival.ui.graph import GraphWindow
from orbital_survival.ui.starfield import create_stars, draw_stars


class Game:
    """
    Main class that manages the entire game system.
    """

    def __init__(self, show_graph: bool = False) -> None:
        """Open the game window, and the graph window if asked.

        Args:
            show_graph: Whether to open the telemetry graph window.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("ORBITAL SURVIVAL")

        # Shared by every round so it does not blink on death. None when closed
        # or without --graph.
        self.graph = GraphWindow() if show_graph else None
        if self.graph is not None:
            self.graph.place_beside_game()

        self.is_running = True
        self.scene = self._create_scene(SceneRequest(GameMode.START))

        self.background_stars = create_stars(NUM_BACKGROUND_STARS)

    def _create_scene(self, request: SceneRequest) -> Scene:
        """Build the scene the request names, handing over its payload.

        Building scenes here keeps them from having to know one another.

        Args:
            request: The mode to switch to, and its payload.

        Returns:
            The new scene.
        """
        match request.mode:
            case GameMode.START:
                return StartScene(self.screen)
            case GameMode.PLAY:
                return PlayScene(self.screen, self.graph)
            case GameMode.GAME_OVER:
                # A missing tally is a caller bug; fail loudly rather than
                # show a score of zero.
                assert request.result is not None
                return GameOverScene(self.screen, request.result)

    def _handle_events(self) -> None:
        """Drain the event queue, letting the scene request a mode switch."""
        for event in pygame.event.get():
            if self._is_graph_close(event):
                self._close_graph()
                continue

            if event.type == pygame.QUIT:
                self.is_running = False

            request = self.scene.handle_event(event)
            if request is not None:
                self.scene = self._create_scene(request)

    def _is_graph_close(self, event: pygame.event.Event) -> bool:
        """Return whether this event is the graph window being closed.

        Checking `event.window` keeps a graph close from ending the game.

        Args:
            event: Event to inspect.

        Returns:
            True if the graph window is being closed.
        """
        if self.graph is None:
            return False
        return event.type == pygame.WINDOWCLOSE and self.graph.owns(
            getattr(event, "window", None)
        )

    def _close_graph(self) -> None:
        """Destroy the graph window and stop drawing to it."""
        if self.graph is not None:
            self.graph.close()
            self.graph = None

    def _update(self) -> None:
        """Advance the current scene one frame, honouring any switch."""
        request = self.scene.update()
        if request is not None:
            self.scene = self._create_scene(request)

    def _draw(self) -> None:
        """Repaint the frame: background, starfield, then the scene."""
        self.screen.fill(BLACK)

        draw_stars(self.screen, self.background_stars)

        self.scene.draw()

        pygame.display.flip()

        # After the flip, so a slow repaint of the graph never delays the
        # frame the player is actually looking at.
        if self.graph is not None:
            self.graph.draw()

    def run(self) -> None:
        """Run the game loop until the window is closed."""
        while self.is_running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(FPS)

        self._close_graph()
        pygame.quit()
