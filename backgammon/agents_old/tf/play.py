import pathlib
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import backgammon.game as bg
from backgammon import agents
from backgammon.agents_old.tf.model import Model
from backgammon.agents_old.tf_agent import TfAgent


class OtherModel(Model):

    LAYER_SIZE_INPUT = 48
    LAYER_SIZE_HIDDEN = 24

    def extract_features(self, board: bg.Board):
        """Create feature to insert in model."""
        def get_features(columns: bg.ColumnCheckersNumber) -> np.ndarray:
            features = np.zeros(board.NUM_COLS)
            for col in range(board.NUM_COLS):
                if col in columns:
                    features[col] = columns[col] / board.NUM_CHECKERS
            return features

        columns, opp_columns = board.to_schema()
        features = np.concatenate((get_features(columns), get_features(opp_columns)))
        return features.reshape(1, -1)


def initializer(path, model_cls):
    def send_inside():
        graph = tf.Graph()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(graph=graph, config=config)
        with sess.as_default():
            with graph.as_default():
                model = model_cls(sess, path, restore=True)
                agent = TfAgent(model)
                return agent
    return send_inside
#
#
PATHS = [
    pathlib.Path('/home/borsden/Projects/russian_backgammon/models/tf') / '1587324086_720_input',
    pathlib.Path('/home/borsden/Projects/russian_backgammon/models/tf') / '1587321691_48_input'
]
#

if __name__ == '__main__':
    ports = [8000, 10000]

    player, opp = [
        agents.TCPAgent.with_server(initializer(path, model_cls), port=port)
        for port, path, model_cls in zip(ports, PATHS, (Model, OtherModel))
    ]

    players = (player, opp)

    time.sleep(1)

    winners = {pl: 0 for pl in players}
    marses = {pl: 0 for pl in players}
    kokses = {pl: 0 for pl in players}

    for episode in tqdm(range(100)):
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
            f"| Mars-{marses[player]}/{marses[opp]}; Koks-{kokses[player]}/{kokses[opp]}"
        )
        players = tuple(reversed(players))
