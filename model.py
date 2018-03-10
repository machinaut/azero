#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf

from game import Game
from util import sample_games


class Model:
    ''' Interface class for a model to be optimized by alphazero algorithm '''

    def __init__(self, n_action, n_view, n_player, seed=None):
        self.rs = np.random.RandomState(seed=seed)
        self.n_act = n_action
        self.n_obs = n_view
        self.n_val = n_player
        self.n_updates = 0

    @classmethod
    def make(cls, game):
        ''' Build an instance from a game '''
        assert isinstance(game, Game), 'Bad game {}'.format(game)
        return cls(game.n_action, game.n_view, game.n_player)

    def model(self, obs):
        '''
        Call the model on a board state
            obs - game state concatenated with current player
        Returns
            logits - action selection probability logits (pre-softmax)
            values - estimated sum of future rewards per player
        '''
        obs = np.asarray(obs, dtype=float)
        assert obs.size == self.n_obs
        logits, values = self._model(obs)
        logits = np.asarray(logits, dtype=float)
        values = np.asarray(values, dtype=float)
        assert logits.size == self.n_act
        assert values.size == self.n_val
        return logits, values

    def _model(self, obs):
        raise NotImplementedError('Implement in subclass')

    def update(self, games):
        '''
        Update model given a list of games.  Each game is a pair of:
            trajectory - list of (obs, probs)
            outcome - total reward per player
        Returns loss (may be evaluated over a subset of game states)
        '''
        self.n_updates += 1
        return self._update(games)

    def _update(self, games):
        # Optionally overwrite this to get dense updates
        # Default is to sample single data point from each game
        # and pass them to _sparse_update().
        obs, q, z = sample_games(games)
        return self._sparse_update(obs, q, z)

    def _sparse_update(self, obs, q, z):
        raise NotImplementedError('Implement this or _update() in subclass')


class Uniform(Model):
    ''' Maximum entropy (uniform distribution) '''

    def _model(self, obs):
        # Fun little hack, sum the observation, then multiply by zero
        # This allows NaN propagation, which is a great way of testing models
        zero = obs.sum() * 0.0
        logits = np.ones(self.n_act) * zero
        values = np.ones(self.n_val) * zero
        return logits, values


class Linear(Model):
    ''' Simple linear model '''

    def __init__(self, *args, scale=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = self.rs.randn(self.n_obs, self.n_act) * scale
        self.V = self.rs.randn(self.n_obs, self.n_val) * scale

    def _model(self, obs):
        obs = obs.flatten()
        logits = obs.dot(self.W)
        values = obs.dot(self.V)
        return logits, values


class Memorize(Model):
    ''' Remember and re-use training data '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {}  # Map from tuple(state) -> (logits, outcome)

    def _model(self, obs):
        ''' Return data if present, else uniform prior '''
        # Hack to ensure NaN propagation
        zero = np.sum(obs) * 0.0
        logits = np.ones(self.n_act) * zero
        values = np.ones(self.n_val) * zero
        return self.data.get(obs.tostring(), (logits, values))


class MLP(Model):
    def __init__(self, *args,
                 hidden_units=[10, 10],
                 drop_rate=0.1,
                 batchnorm=True,
                 activation=tf.nn.relu,
                 learning_rate=0.001,
                 combination=0.5,
                 step_update=1,
                 step_trace=1,
                 step_save=1,
                 step_summary=1,
                 save_path=None,
                 log_dir='/tmp/azero',
                 **kwargs):
        '''
        Build simple fully-connected network.
            hidden_units - list of sizes of hidden layers
            drop_rate - dropout rate (set to 0.0 to disable)
            batchnorm - boolean to enable batchnorm
            activation - which tensorflow activation function to use
            learning_rate - optimization step size
            combination - linear combination of loss terms
            step_update - how many steps of optimization per batch
            step_trace - how many steps between full traces
            step_save - how many steps between saving model
            step_summary - how many steps between writing summary
            save_path - path to save the model checkpoints
            log_dir - directory to save event log for tensorboard
        Returns:
            obs - observation input placeholder
            act - action output tensor
        '''
        super().__init__(*args, **kwargs)
        self.step_update = step_update
        self.step_trace = step_trace
        self.step_save = step_save
        self.step_summary = step_summary
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'model.ckpt')
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Set to True when we're training
        self.training = tf.placeholder(tf.bool, (), name='training')
        tf.add_to_collection('training', self.training)

        # input layer
        self.obs = tf.placeholder(tf.float32, [None, self.n_obs], name='obs')
        tf.add_to_collection('obs', self.obs)
        net = tf.identity(self.obs)
        # hidden layers
        for i, units in enumerate(hidden_units):
            # TODO: experiment with activation before/after other layers?
            net = tf.layers.dense(net, units=units, name='dense%d' % i)
            if batchnorm:
                net = tf.layers.batch_normalization(net, training=self.training,
                                                    name='batchnorm%d' % i)
            net = tf.layers.dropout(net, rate=drop_rate,
                                    training=self.training,
                                    name='dropout%d' % i)
            if activation is not None:
                net = activation(net, name='activation%d' % i)
        # output layers
        self.p = tf.layers.dense(net, self.n_act, name='p')
        tf.add_to_collection('p', self.p)
        self.v = tf.layers.dense(net, self.n_val, name='v')
        tf.add_to_collection('v', self.v)

        # placeholders for input
        self.q = tf.placeholder(tf.float32, [None, self.n_act], name='q')
        tf.add_to_collection('q', self.q)
        self.z = tf.placeholder(tf.float32, [None, self.n_val], name='z')
        tf.add_to_collection('z', self.z)

        # Loss terms
        with tf.name_scope('loss'):
            combination = tf.constant(combination, dtype=tf.float32, name='c')
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.q, logits=self.p)
            mse = tf.reduce_mean(tf.square(self.v - self.z), axis=1)
            self.loss = tf.reduce_mean(
                xent * combination + mse * (1 - combination))
            tf.add_to_collection('loss', self.loss)
            tf.summary.scalar('loss', self.loss)

        # Set global step tensor if not already set
        if tf.train.get_global_step() is None:
            tf.Variable(0, trainable=False, name='global_step')

        # Optimizer
        optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = optimizer_op.minimize(
                self.loss, global_step=tf.train.get_global_step(),
                name='train')
        tf.add_to_collection('train', self.train)

        # Make our session and initialize our variables
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Summary and tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer.add_graph(self.sess.graph)

        # Saver for model checkpoints
        self.saver = tf.train.Saver()

    def _model(self, obs):
        assert obs.size == self.n_obs, 'bad obs size {}'.format(obs)
        feed_dict = {self.obs: obs.flatten()[None, :],  # Add batch dimension
                     self.training: False}
        p, v = self.sess.run([self.p, self.v], feed_dict=feed_dict)
        return p[0], v[0]  # Remove batch dimension

    def _sparse_update(self, obs, q, z):
        global_step = tf.train.get_global_step()
        assert global_step is not None, 'Missing global step tensor!'
        for _ in range(self.step_update):
            i = tf.train.global_step(self.sess, global_step)
            print('step:', i)

            # Optionally run a full trace to generate graph info
            if i % self.step_trace == self.step_trace - 1:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            # Generate dataset to train on
            feed_dict = {self.obs: obs.reshape(obs.shape[0], -1), self.q: q,
                         self.z: z, self.training: True}
            if i % self.step_summary == self.step_summary - 1:
                summary, loss, _ = self.sess.run([self.merged, self.loss,
                                                  self.train],
                                                 feed_dict=feed_dict,
                                                 options=run_options,
                                                 run_metadata=run_metadata)
                self.writer.add_summary(summary, global_step=i)
            else:
                loss, _, = self.sess.run([self.loss, self.train], feed_dict=feed_dict,
                                         options=run_options, run_metadata=run_metadata)

            # If we generated a trace, write it out
            if run_metadata is not None:
                self.writer.add_run_metadata(run_metadata, 'step%d' % i)
                print('Saved metadata for step:', i)

            # Save a model checkpoint
            if i % self.step_save == self.step_save - 1:
                saved_path = self.saver.save(
                    self.sess, self.save_path, global_step=i)
                print('Model saved in path:', saved_path)


models = [Uniform, Linear, Memorize, MLP]


if __name__ == '__main__':
    n_obs, n_act, n_val = 5, 3, 1
    model = MLP(n_action=n_act, n_view=n_obs, n_player=n_val)
    batch_size = 4
    obs = np.random.randn(batch_size, n_obs)
    q = np.random.randn(batch_size, n_act)
    z = np.random.randn(batch_size, n_val)
    model._sparse_update(obs, q, z)
