import asyncio
import json
import multiprocessing
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

    def extract_features(self, board: bg.Board) -> torch.Tensor:
        """Create feature to insert in model.
        Generate array of 720 features, 15 features for every position and same for opponent.
        """

        def get_features(columns: bg.ColumnCheckersNumber) -> np.ndarray:
            features = np.zeros(board.NUM_COLS * board.NUM_CHECKERS)
            for col in range(board.NUM_COLS):
                if col in columns:
                    start = col * board.NUM_CHECKERS
                    end = start + columns[col]
                    features[start:end] = 1
            return features

        columns, opp_columns = board.to_schema()
        features = np.concatenate((get_features(columns), get_features(opp_columns)))
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


class TCPAgent(bg.Agent):
    def get_action(self, available_moves: List[bg.Moves], board: bg.Board) -> bg.Moves:
        """Send a message to the server, wait an answer and use it."""

        async def tcp_echo_client(message):
            reader, writer = await asyncio.open_connection(self.host, self.port)
            writer.write(message.encode())
            data = await reader.read(100000)
            writer.close()
            return json.loads(data.decode())

        message = json.dumps(dict(available_moves=available_moves, board=board.to_schema()))
        done = asyncio.run(tcp_echo_client(message))
        return done

    def __init__(self, host: str = None, port: int = None, agent_name: str = None):
        self.host = host
        self.port = port
        self.agent_name = agent_name

    def __repr__(self):
        information = ''
        if self.host:
            information += f'[{self.host}]'
        if self.port:
            information += f'[:{self.port}]'
        if self.agent_name:
            information += f'[{self.agent_name}]'

        return f'{self.__class__.__name__}{information}'

    @classmethod
    def with_server(
        cls,
        agent_initializer: Callable[[], bg.Agent],
        port: int = None, host: str = None
    ) -> 'TCPAgent':
        """Run server in child process, return insta"""
        if not host and not port:
            raise ValueError('Should specified at least host or port.')

        pipe = multiprocessing.Pipe(False)
        proc = multiprocessing.Process(
            target=cls._server_runner,
            args=(agent_initializer,),
            kwargs=dict(port=port, host=host, pipe=pipe)
        )
        proc.start()

        pipe_out, _ = pipe
        agent_name = pipe_out.recv()

        return cls(
            host=host, port=port, agent_name=agent_name
        )

    @classmethod
    def _server_runner(
        cls,
        agent_initializer: Callable[[], bg.Agent],
        host: str = None,
        port: int = None,
        pipe: multiprocessing.Pipe = None
    ) -> None:
        """Create a TCP server, which can receive board and available values and select an action.

        :param agent_initializer: function to initialize Agent. Do not pass Agent instance directly, because there are
        situations, where we should generate it already in another process.
        :param host: host
        :param port: port
        :param pipe: Pipe. Send name of created agent, if specified.
        """

        async def handle(reader, writer):
            data = await reader.read(100000)
            message = json.loads(data.decode())
            move = agent.get_action(
                available_moves=message['available_moves'],
                board=bg.Board.from_schema(*message['board'])
            )
            writer.write(json.dumps(move).encode())
            await writer.drain()
            writer.close()

        async def run_server():
            server = await asyncio.start_server(handle, host, port)
            async with server:
                await server.serve_forever()

        agent = agent_initializer()

        if pipe:
            _, pipe_in = pipe
            pipe_in.send(str(agent))

        asyncio.run(run_server())
