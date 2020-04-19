import os
import time

import tensorflow as tf

from backgammon.agents.tf.model import Model, Model2

path = os.path.join(
    'backgammon/agents/tf/models/',
    '1541165560_float_input/'
)

ACTION = 'test'


if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            model = Model2(sess, path, restore=True)
            model.test(episodes=1)
