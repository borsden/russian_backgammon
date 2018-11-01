from typing import Set
import backgammon.game as bg


class TfAgent(bg.Agent):
    """Tensorflow td-gammon player."""

    def __init__(self, model: 'Model'):
        self.model = model

    def get_action(self, available_moves: Set[bg.Moves], game: bg.Game) -> bg.Moves:
        """Find and return best action."""
        v_best = 0
        best_moves = None

        for moves in available_moves:
            with game.board.temp_move(*moves):
                features = self.model.extract_features(game)
                v = self.model.get_output(features)
                if v > v_best:
                    v_best = v
                    best_moves = moves

        return best_moves
