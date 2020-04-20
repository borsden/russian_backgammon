from pprint import pprint
from typing import Set
import backgammon.game as bg


class TfAgent(bg.Agent):
    """Tensorflow td-gammon player."""

    def __init__(self, model):
        self.model = model

    def get_action(self, available_moves: Set[bg.Moves], board: bg.Board) -> bg.Moves:
        """Find and return best action."""
        v_best = 0
        best_moves = None

        for moves in available_moves:
            try:
                with board.temp_move(*moves):
                    features = self.model.extract_features(board)
                    v = self.model.get_output(features)
                    if v > v_best:
                        v_best = v
                        best_moves = moves
            except Exception as e:
                print(board)
                print(board.to_schema())
                print(available_moves)
                print(moves)
                raise e

        # pprint(__res)
        return best_moves

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.model.path.name}]'
