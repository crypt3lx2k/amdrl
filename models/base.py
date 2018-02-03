#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import spec
from . import model_fns
from . import train_fns

import tensorflow as tf

def build_placeholder (data_config, name=None):
    """Creates a placeholder based on a DataConfig."""
    placeholder = tf.placeholder (
        dtype=data_config.dtype,
        shape=(None,) + data_config.shape,
        name=name
    )
    
    return placeholder

def fill_dict (feed_dict, placeholders, data):
    """Feeds a dictionary of data into a dictionary of placeholders."""
    for k in data:
        feed_dict[placeholders[k]] = data[k]

    return feed_dict

class Model (object):
    """Model base class."""
    def __init__ (self, input_config, output_config):
        self.input_config = input_config
        self.output_config = output_config

class TFModel (Model):
    """Base class for all TensorFlow based models."""
    def __init__ (
            self,
            input_config, output_config,
            params
    ):
        """."""
        super(TFModel, self).__init__ (
            input_config=input_config,
            output_config=output_config
        )

        self.params = params
        self.graph = tf.Graph()

    def __enter__ (self):
        self.contexts = []

        # Add context managers
        self.contexts.append(self.graph.as_default())
        self.contexts.append(self.session)

        # Enter contexts in order
        ctx_values = map(lambda ctx : ctx.__enter__(), self.contexts)
        return self

    def __exit__ (self, ex_type, ex_val, ex_trace):
        suppress = False

        # Pop contexts in reverse order as they were entered
        while self.contexts:
            ctx = self.contexts.pop()
            # Exit context
            ctx_suppress = ctx.__exit__(ex_type, ex_val, ex_trace)

            # Suppress if any of our managers decided to handle the error
            suppress = suppress or ctx_suppress

        return suppress

    # FIXME: Figure out where this entire build_server method belongs
    def build_server (self):
        if self.cluster is None:
            return tf.train.Server.create_local_server (
                config=self.config
            )

        self.cluster = tf.train.ClusterSpec(self.cluster)
        return tf.train.Server (
            self.cluster,
            job_name='worker',
            task_index=self.params.get('task_index'),
            config=self.config
        )

    def build (self, input_fn, inference_fn, loss_fn):
        """Builds entire computation graph with session."""
        optimizer_fn, optimizer_params = train_fns.get_optimizer_fn(self.params)
        train_fn = train_fns.make_train_fn(optimizer_fn, optimizer_params)
        model_fn = model_fns.make_model_fn(inference_fn, loss_fn, train_fn)

        # Set up session config if available
        self.config = self.params.get('config')
        if self.config is not None:
            self.config = tf.ConfigProto(**self.config)

        # Get cluster specification and launch server
        self.cluster = self.params.get('cluster')
        self.server = self.build_server()

        # Build graph
        with self.graph.as_default():
            with tf.device (
                    tf.train.replica_device_setter (
                        worker_device='/job:worker/task:{}'.format (
                            self.params.get('task_index')
                        ),
                        cluster=self.cluster
                    )
            ):
                # Create inputs
                with tf.variable_scope('inputs'):
                    self.features, self.labels = input_fn()

                # Create training model
                with tf.variable_scope('model'):
                    self.global_step = tf.train.create_global_step()

                    self.train_specification = model_fn (
                        features=self.features,
                        labels=self.labels,
                        mode=spec.Modes.TRAIN,
                        params=self.params
                    )

                # Create inference model
                with tf.variable_scope('model', reuse=True):
                    self.predict_specification = model_fn (
                        features=self.features,
                        labels=self.labels,
                        mode=spec.Modes.PREDICT,
                        params=self.params
                    )

                # Create session
                with tf.variable_scope('session'):
                    self.session = tf.train.MonitoredTrainingSession (
                        master=self.server.target,
                        checkpoint_dir=self.params['model_dir'],
                        is_chief=self.params.get('is_chief'),
                        config=self.config
                    )

        return self

    def predict (self, features):
        """Returns prediction based on input features."""
        feed_dict = fill_dict({}, self.features, features)

        return self.session.run (
            self.predict_specification.predictions,
            feed_dict=feed_dict
        )

    def train (self, features, labels):
        """Runs training operation."""
        feed_dict = {}
        feed_dict = fill_dict(feed_dict, self.features, features)
        feed_dict = fill_dict(feed_dict, self.labels, labels)

        loss, _ = self.session.run (
            [
                self.train_specification.loss,
                self.train_specification.train_op
            ],
            feed_dict=feed_dict
        )

        return loss

    def step (self):
        """Returns training step."""
        return self.session.run(self.global_step)
