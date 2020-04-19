import os
import time

import tensorflow as tf

from backgammon.agents_old.tf.model import Model



path = os.path.join(
    '/home/borsden/Projects/russian_backgammon/models/tf',
    f'{str(int(time.time()))}_720_input_with_mars_and_koks'
)


if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            model = Model(sess, path, restore=False)
            model.train()
