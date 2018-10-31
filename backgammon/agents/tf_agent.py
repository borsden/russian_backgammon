from typing import Set
import backgammon.game as bg


class TfAgent(bg.Agent):
    """Tensorflow td-gammon player."""

    def __init__(self, model: 'Model'):
        self.model = model

    def get_action(self, available_moves: Set[bg.Moves], game: bg.Game) -> bg.Moves:
        """Find and return best action."""
        v_best = 0
        a_best = None

        for move in available_moves:
            return move
            # ateList = game.take_action(a, self.player)
            # features = game.extract_features(game.opponent(self.player))
            # v = self.model.get_output(features)
            # v = 1. - v if self.player == game.players[0] else v
            # if v > v_best:
            #     v_best = v
            #     a_best = a
            # game.undo_action(a, self.player, ateList)

        # return a_best