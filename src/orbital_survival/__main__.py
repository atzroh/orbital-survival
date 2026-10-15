"""Entry point for `orbital-survival` and `python -m orbital_survival`."""

import argparse

from orbital_survival.game import Game


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="orbital-survival",
        description="A 2D space survival game: orbit your planet and dodge beams.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help=(
            "plot speed, acceleration and input over time in a second window "
            "(close it at any time; the game keeps running)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Start the game and run it until the window is closed."""
    Game(show_graph=parse_args().graph).run()


if __name__ == "__main__":
    main()
