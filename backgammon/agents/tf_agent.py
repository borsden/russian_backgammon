from pprint import pprint
from typing import Set
import backgammon.game as bg


class TfAgent(bg.Agent):
    """Tensorflow td-gammon player."""

    def __init__(self, model: 'Model'):
        self.model = model

    def get_action(self, available_moves: Set[bg.Moves], board: bg.Board) -> bg.Moves:
        """Find and return best action."""
        v_best = 0
        best_moves = None
        # __res = {}

        for moves in available_moves:
            with board.temp_move(*moves):
                features = self.model.extract_features(board)
                v = self.model.get_output(features)
                # __res[moves] = v[0][0]
                if v > v_best:
                    v_best = v
                    best_moves = moves

        # pprint(__res)
        return best_moves
