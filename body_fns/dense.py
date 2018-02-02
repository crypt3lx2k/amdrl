#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

__all__ = ['make_body_fn']

def make_body_fn (hidden_units, dropout_rate):
    """Makes a fully connected TF body model function."""

    def body_fn (features, training=False):
        """Returns a fully connected body model."""
        # Flatten input features
        flat_features = tf.layers.flatten(features, name='features/flattened')

        # Hidden layers with dropout
        net = flat_features
        for i, units in enumerate(hidden_units):
            net = tf.layers.dense (
                inputs=net,
                units=units,
                # FIXME: Different activation functions
                activation=tf.nn.tanh,
                name='hidden_{}/dense_{}'.format(i, units)
            )

            if training:
                net = tf.layers.dropout (
                    inputs=net,
                    rate=dropout_rate,
                    training=training,
                    name='hidden_{}/dropout'.format(i)
                )

        return net

    return body_fn
