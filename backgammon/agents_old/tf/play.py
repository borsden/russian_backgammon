import pathlib
import time
from multiprocessing import Process

import tensorflow as tf
from tqdm import tqdm

from backgammon import agents
from backgammon.agents_old.tf.model import Model
from backgammon.agents_old.tf_agent import TfAgent
import backgammon.game as bg


def initializer(path):
    def send_inside():
        graph = tf.Graph()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(graph=graph, config=config)
        with sess.as_default():
            with graph.as_default():
                model = Model(sess, path, restore=True)
                agent = TfAgent(model)
                return agent
    return send_inside
#
#
PATHS = [
    pathlib.Path('/home/borsden/Projects/russian_backgammon/models/tf') / '1587324086_720_input',
    pathlib.Path('/home/borsden/Projects/russian_backgammon/models/tf') / '1587326248_720_input_mars_koks'
]
#

if __name__ == '__main__':
    ports = [8000, 10000]

    player, opp = [
        agents.TCPAgent.with_server(initializer(path), port=port)
        for port, path in zip(ports, PATHS)
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
