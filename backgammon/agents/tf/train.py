import os
import time

import tensorflow as tf

from backgammon.agents.tf.model import Model2



path = os.path.join(
    'backgammon/agents/tf/models/',
    '{}_float_input_with_mars_and_koks'.format(str(int(time.time())))
)


if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            # model = Model2(sess, path, restore=True)
            model = Model2(sess, path, restore=False)
            model.train()
