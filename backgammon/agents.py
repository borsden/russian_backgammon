import random
from functools import partial
from typing import Set, Callable, List, Iterator

import numpy as np
import torch
from torch import nn

import backgammon.game as bg


class RandomAgent(bg.Agent):
    """Random Player."""

    def get_action(self, available_moves: List[bg.Moves], board: bg.Board) -> bg.Moves:
        return random.choice(list(available_moves))


class NNAgent(bg.Agent):
    """Neural network player."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        """Model, which can predict a quality of state."""
    #
    # def extract_features(self, board: bg.Board) -> torch.Tensor:
    #     """Create feature to insert in model.
    #     Generate array of 720 features, 15 features for every position and same for opponent.
    #     """
    #     def get_features(columns: bg.ColumnCheckersNumber) -> np.ndarray:
    #         features = np.zeros(board.NUM_COLS * board.NUM_CHECKERS)
    #         for col in range(board.NUM_COLS):
    #             if col in columns:
    #                 start = col * board.NUM_CHECKERS
    #                 end = start + columns[col]
    #                 features[start:end] = 1
    #         return features
    #
    #     columns, opp_columns = board.to_schema()
    #     features = np.concatenate((get_features(columns), get_features(opp_columns)))
    #     return torch.from_numpy(features).float().cuda()
    def extract_features(self, board: bg.Board) -> torch.Tensor:
        """Create feature to insert in model.
        Generate array of 24 features, 15 features for every position and same for opponent.
        """
        columns, opp_columns = board.to_schema(straight=True)
        features = np.zeros(board.NUM_COLS)
        for col in columns:
            features[col] = columns[col]
        for col in opp_columns:
            features[col] = - opp_columns[col]
        return torch.from_numpy(features).float().cuda()

    def estimate_moves(self, available_moves: List[bg.Moves], board: bg.Board) -> Iterator[float]:
        """Estimate resulting board position for all passed moves."""
        for moves in available_moves:
            with board.temp_move(*moves) as temp_board:
                v = self.estimate(temp_board)
                yield v

    def get_action(self, available_moves: List[bg.Moves], board: bg.Board) -> bg.Moves:
        """Find and return best action."""
        available_moves = list(available_moves)
        estimated_moves = list(self.estimate_moves(available_moves, board))
        index_max = int(np.argmin(estimated_moves))

        return available_moves[index_max]

    def estimate(self, board):
        """Get a value of specified position."""
        features = self.extract_features(board)
        v = self.model(features)
        return v

    def __repr__(self):
        return f'{self.__class__.__name__}[model={self.model}]'

    @classmethod
    def with_model_constructor(cls, model: nn.Module) -> Callable[[], 'NNAgent']:
        """
        Create a child of current class with specified model.
        :param model: torch model
        :return: NNAgent class with specified model
        """
        return partial(cls, model=model)
