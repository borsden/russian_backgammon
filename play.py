import time

import backgammon.game as bg
from backgammon.agents.random_agent import RandomAgent

# if __name__ == '__main__':
#     # board = Board()
#     t1 = time.time()
#     players = [RandomAgent(), RandomAgent()]
#     game = bg.Game(players=players, logs=False)
#     winner = game.play()
#     game.board.draw()
#     print(winner.checker_type)
#     print(time.time() - t1)
#
#
import os
import tensorflow as tf

from backgammon.agents.tf.model import Model


model_path = os.environ.get('MODEL_PATH', 'backgammon/agents/tf/models/')
summary_path = os.environ.get('SUMMARY_PATH', 'backgammon/agents/tf/summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'backgammon/agents/tf/checkpoints/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)


# ACTION = 'test'
ACTION = 'train'


if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            model = Model(sess, model_path, summary_path, checkpoint_path, restore=False)

            if ACTION == 'train':
                model.train()
            elif ACTION == 'test':
                model.test()
