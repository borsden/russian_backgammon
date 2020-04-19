from typing import List

import itertools
import os
import pathlib
import time

import numpy as np
import tensorflow as tf

# from backgammon.agents.human_agent import HumanAgent
from tqdm import tqdm

from backgammon.agents import RandomAgent
from backgammon.agents_old.tf_agent import TfAgent
import backgammon.game as bg


# helper to initialize a weight and bias variable
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
    return W, b


# helper to create a dense, fully-connected layer
def dense_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(shape)
        return activation(tf.matmul(x, W) + b, name='activation')


class Model:

    LAYER_SIZE_INPUT = 720
    LAYER_SIZE_HIDDEN = 100

    def __init__(self, sess: tf.Session, path: str, restore: bool = False) -> None:
        """Create tensorflow model. Write graph"""

        self.path = path
        self.checkpoint_path = os.path.join(path, 'checkpoints')
        self.summaries_path = os.path.join(path, 'summaries')

        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.summaries_path).mkdir(parents=True, exist_ok=True)

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay
        lamda = tf.maximum(
            0.7,
            tf.train.exponential_decay(0.9, self.global_step, 30000, 0.96, staircase=True),
            name='lambda'
        )

        # learning rate decay
        alpha = tf.maximum(
            0.01,
            tf.train.exponential_decay(0.1, self.global_step, 40000, 0.96, staircase=True),
            name='alpha'
        )

        tf.summary.scalar('lambda', lamda)
        tf.summary.scalar('alpha', alpha)

        # describe network size
        layer_size_input = self.LAYER_SIZE_INPUT
        layer_size_hidden = self.LAYER_SIZE_HIDDEN
        layer_size_output = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [1, layer_size_output], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, [layer_size_input, layer_size_hidden], tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, [layer_size_hidden, layer_size_output], tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.summary.scalar('V_next', tf.reduce_sum(self.V_next))
        tf.summary.scalar('V', tf.reduce_sum(self.V))

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'),
                                    name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)

            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)
            accuracy_avg_op = accuracy_sum / tf.maximum(game_step, 1.0)

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])

            tf.summary.scalar('game/loss_avg', loss_avg_op)
            tf.summary.scalar('game/delta_avg', delta_avg_op)
            tf.summary.scalar('game/accuracy_avg', accuracy_avg_op)
            tf.summary.scalar('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))
            tf.summary.scalar('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))
            tf.summary.scalar('game/accuracy_avg_ema', accuracy_avg_ema.average(accuracy_avg_op))

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)
                    tf.summary.histogram(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                grad_trace = alpha * delta_op * trace_op
                tf.summary.histogram(var.name + '/gradients/trace', grad_trace)

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            accuracy_sum_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
            accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.summary.merge_all()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=10)

        # run variable initializers
        self.sess.run(tf.global_variables_initializer())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

    def restore(self):
        """Restore latest checkpoint."""
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print(('Restoring checkpoint: {0}'.format(latest_checkpoint_path)))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={self.x: x})

    #
    # def play(self):
    #     game = Game.new()
    #     game.play([TDAgent(Game.TOKENS[0], self), HumanAgent(Game.TOKENS[1])], draw=True)
    #
    def extract_features(self, board: bg.Board) -> np.ndarray:
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
        return features.reshape(1, -1)

    def test(self, episodes=100):
        player, opp = (TfAgent(self), RandomAgent())
        players = (player, opp)

        winners = {pl: 0 for pl in players}
        marses = {pl: 0 for pl in players}
        kokses = {pl: 0 for pl in players}

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
                f"[Episode {episode}] {player.name} vs {opp.name} "
                f"{winners[player]}:{winners[opp]} of {winners_total} games "
                f"- ({(winners[player] / winners_total) * 100.0:.2f})% "
                f"| Mars-{marses[player]}/{marses[opp]}; Koks-{kokses[player]}/{kokses[opp]}"
            )
            players = tuple(reversed(players))

    def train(self):
        tf.train.write_graph(self.sess.graph, self.path, 'model.pb', as_text=False)
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.summaries_path, str(int(time.time()))),
            self.sess.graph
        )

        # the agent plays against itself, making the best move for each player
        players = (TfAgent(self), TfAgent(self))

        validation_interval = 500
        episodes = 5000

        for episode in range(episodes):
            game = bg.Game(players=players)

            game_step = 0

            x = self.extract_features(game.board)

            is_opp = True

            while game.board.status is None:
                current_player = next(game.players_steps)
                with game.board.reverse(fake=not is_opp) as board:
                    game.make_step(player=current_player, board=board)
                    x_next = self.extract_features(game.board)
                    V_next = self.get_output(x_next)

                self.sess.run(self.train_op, feed_dict={self.x: x, self.V_next: V_next})
                x = x_next
                game_step += 1

            V_next = game.board.status

            _, global_step, summaries, _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.summaries_op,
                self.reset_op
            ], feed_dict={self.x: x, self.V_next: np.array([[V_next]], dtype='float')})

            summary_writer.add_summary(summaries, global_step=global_step)
            winner = 'X' if current_player == players[0] else 'O'
            print(f"Game {episode}/{episodes} (Winner: {winner}) in {game_step} turns")

            if (episode + 1) % validation_interval == 0:
                self.saver.save(self.sess, os.path.join(self.checkpoint_path, 'checkpoint'), global_step=global_step)
                self.test(episodes=100)

        summary_writer.close()

        self.test(episodes=1000)


class Model2(Model):
    LAYER_SIZE_INPUT = 50
    LAYER_SIZE_HIDDEN = 25

    @classmethod
    def extract_features(self, board: bg.Board) -> List[List[float]]:
        """Create feature to insert in model."""
        def _get_features(opponent: bool=False) -> List[float]:
            positions_with_count = {
                pos: len(board.cols[pos])
                for pos in board.get_occupied_positions(opponent)
            }
            outed_checkers_percent = sum(positions_with_count.values()) / board.NUM_CHECKERS

            _features = [0 for _ in range(board.NUM_COLS)]

            for pos, count in positions_with_count.items():
                _features[pos] = count/board.NUM_CHECKERS

            return _features + [outed_checkers_percent]


        features = _get_features() + _get_features(opponent=True)
        return np.array(features).reshape(1, -1)
