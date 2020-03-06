import random
from typing import Set

import backgammon.game as bg


class RandomAgent(bg.Agent):
    """Random Player."""

    def get_action(self, available_moves: Set[bg.Moves], board: bg.Board) -> bg.Moves:
        return random.choice(list(available_moves))
