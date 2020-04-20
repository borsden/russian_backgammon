import pathlib
import random
import sys
from contextlib import contextmanager
from os import PathLike
from typing import List, Type, Callable
from unittest import mock

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

import backgammon.game as bg
from backgammon import agents


# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter('%(message)s'))
#
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)
#

class Trainer:
    """Object, that can train specified model, using it learning algorithm."""

    def __init__(
            self,
            model: torch.nn.Module,
            agent_cls: Type[agents.NNAgent] = agents.NNAgent,
            loss: torch.nn.Module = torch.nn.MSELoss(),
            optimizer_func: Callable[..., Optimizer] = lambda params: torch.optim.Adam(params, lr=0.05)
    ):
        """Initialize trainer
        :param model: torch model, which predict Q for specified board
        :param agent_cls: agent class
        :param loss: loss function
        :param optimizer_func: function, which receive parameters of model and return torch Optimizer
        """
        self.model = model.cuda()
        self.agent_cls = agent_cls
        self.loss = loss

        self.optimizer = optimizer_func(self.model.parameters())

    def save(self, path: pathlib.Path, episode: int) -> None:
        """Save model onto disk."""
        torch.save(model.state_dict(), path / f"{episode}.pth")

    def validate(self, episodes: int = 100, path: pathlib.Path = None) -> None:
        """Compare current model against random player, save it on disk.
        :param episodes: number of games to play against random player.
        :param path: if specified, log into it.
        """
        @contextmanager
        def validate_log_open(path: pathlib.Path = None):
            """Open validate file or write into stdout, if it is not specified."""
            if path:
                path = path / 'validate.log'
                path.touch(exist_ok=True)
                with open(path, 'a') as f:
                    yield f
            else:
                yield sys.stdout

        self.model.eval()
        player, opp = self.agent_cls(self.model), agents.RandomAgent()
        players = (player, opp)

        winners = {pl: 0 for pl in players}
        marses = {pl: 0 for pl in players}
        kokses = {pl: 0 for pl in players}

        with validate_log_open(path) as f:
            for episode in tqdm(range(episodes)):
                game = bg.Game(players=players, show_logs=False)

                winner, status = game.play()

                winners[winner] += 1

                if status == 2:
                    marses[winner] += 1
                elif status == 3:
                    kokses[winner] += 1

                winners_total = sum(list(winners.values()))
                tqdm.write(
                    f"[Episode {episode}] {player} vs {opp} "
                    f"{winners[player]}:{winners[opp]} of {winners_total} games "
                    f"- ({(winners[player] / winners_total) * 100.0:.2f})% "
                    f"| Mars-{marses[player]}/{marses[opp]}; Koks-{kokses[player]}/{kokses[opp]}",
                    file=f
                )
                players = tuple(reversed(players))
            tqdm.write('____________________________________________________________', file=f)

    @contextmanager
    def e_greedy_get_action(self, episode: int, epsilon: float = 0.) -> None:
        """
        By default agent `get_action` is greedy. It means, that we get a move, which maximize board state.
        Instead of it, we place this behavior with ε-greedy algorithm - with epsilon chance to get a random move.

        Use it as a contextmanager, because want return default behavior after that.
        :param episode: current episode
        :param epsilon: chance to get random move
        """

        def wrapped_get_action(agent: agents.NNAgent, available_moves: List[bg.Moves], board: bg.Board) -> bg.Moves:
            if random.random() < epsilon:
                return random.choice(available_moves)
            return raw_get_action(agent, available_moves, board)

        raw_get_action = self.agent_cls.get_action
        self.agent_cls.get_action = wrapped_get_action
        yield
        self.agent_cls.get_action = raw_get_action

    def update(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> None:
        """Update weights of model given a training example.

        :param y_pred: predicted value
        :param y_target: target value
        """
        if y_target > 10:
            y_target
        loss = self.loss(y_pred.cuda(), Variable(y_target.cuda()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(
            self, episodes: int,
            validation: int = None,
            save_path: PathLike = None,
            gamma: float = 1.0, epsilon: float = 0.2,
    ) -> None:
        """

        :param episodes:
        :param validation:
        :param save_path: path to save models and logs.
        :param gamma: γ - discount factor. Is used to balance immediate and future reward.
        :param epsilon: ε - chance to get random move in ε-greedy policy
        """
        if save_path:
            save_path = pathlib.Path(save_path) / f'SIGMOID_24_negative_reward_gamma_{gamma}_epsilon_{epsilon}_q_learning'
            save_path.mkdir(exist_ok=False, parents=False)

        for episode in tqdm(range(episodes)):
            if validation is not None and not (episode + 1) % validation:
                self.validate(path=save_path)
                if save_path:
                    self.save(save_path, episode + 1)

            self.model.train()

            players = (self.agent_cls(self.model), self.agent_cls(self.model))
            game = bg.Game(players=players)

            with self.e_greedy_get_action(episode, epsilon=epsilon):
                for agent, new_board, prev_board, move, available_moves in game.play_step_by_step():
                    agent: agents.NNAgent
                    pred_q = agent.estimate(board=prev_board)
                    if new_board.status:
                        reward = new_board.status
                        self.update(pred_q, torch.Tensor([reward]))
                        with prev_board.reverse() as reversed_board:
                            self.update(agent.estimate(board=reversed_board), torch.Tensor([-reward]))
                        break
                    else:
                        estimated_moves = list(agent.estimate_moves(available_moves=available_moves, board=prev_board))
                        agent_checkers, opp_checkers = prev_board.to_schema()

                        if estimated_moves:
                            max_q = np.max(estimated_moves)
                            new_q = gamma * max_q
                        else:
                            # it is too bad, if we could not make any step.
                            new_q = torch.Tensor([-1])

                        self.update(pred_q, new_q)


def play(episodes=100):
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(24, 100),
    #     # torch.nn.Linear(720, 100),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(100, 1)
    # ).cuda()
    # model.load_state_dict(torch.load(pathlib.Path('./models/720_gamma_0.95_epsilon_0.3_q_learning/1500.pth')))
    # model.eval()
    #
    # player, opp = agents.NNAgent(model), agents.RandomAgent()
    player, opp = agents.RandomAgent(), agents.RandomAgent()
    players = (player, opp)

    winners = {pl: 0 for pl in players}
    marses = {pl: 0 for pl in players}
    kokses = {pl: 0 for pl in players}

    for episode in tqdm(range(episodes)):
        game = bg.Game(players=players, show_logs=True)

        winner, status = game.play()

        winners[winner] += 1

        if status == 2:
            marses[winner] += 1
        elif status == 3:
            kokses[winner] += 1

        winners_total = sum(list(winners.values()))
        tqdm.write(
            f"[Episode {episode}] {player} vs {opp} "
            f"{winners[player]}:{winners[opp]} of {winners_total} games "
            f"- ({(winners[player] / winners_total) * 100.0:.2f})% "
            f"| Mars-{marses[player]}/{marses[opp]}; Koks-{kokses[player]}/{kokses[opp]}",
            file=None
        )
        # players = tuple(reversed(players))
    # tqdm.write('____________________________________________________________', file=None)


def check():
    model = torch.nn.Sequential(
        torch.nn.Linear(24, 40),
        # torch.nn.Linear(720, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 1)
    ).cuda()
    model.load_state_dict(
        torch.load(pathlib.Path('./models/24_negative_reward_gamma_0.9_epsilon_0.3_q_learning/1500.pth')))
    model.eval()
    player, opp = agents.NNAgent(model), agents.RandomAgent()
    players = (player, opp)
    game = bg.Game(players=players, show_logs=True)
    game.board = bg.Board.from_schema( {22: 11, }, {23: 5, 22: 4})

    available_moves = game.board.get_available_moves((1, 1))
    result = list(player.estimate_moves(available_moves, game.board))
    result


if __name__ == '__main__':
    # play()
    # check()
    model = torch.nn.Sequential(
        torch.nn.Linear(24, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 1),
        torch.nn.Sigmoid(),

    )
    Trainer(model).train(10000, 500, epsilon=0.3, gamma=0.9, save_path=pathlib.Path('./models'))
