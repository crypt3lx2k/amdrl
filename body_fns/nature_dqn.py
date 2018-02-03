#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

__all__ = ['make_body_fn']

def make_body_fn ():
    """Makes the body function used in the DQN nature paper."""

    def body_fn (features, training=False):
        """Returns a body model."""
        net = features

        # 
        net = tf.layers.conv2d (
            inputs=net,
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation=tf.nn.relu,
            name='hidden_0/conv_32_8x8_4x4_relu'
        )

        # 
        net = tf.layers.conv2d (
            inputs=net,
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation=tf.nn.relu,
            name='hidden_1/conv_64_4x4_2x2_relu'
        )

        # 
        net = tf.layers.conv2d (
            inputs=net,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.relu,
            name='hidden_2/conv_64_3x3_1x1_relu'
        )

        # 
        net = tf.layers.flatten(net, name='hidden_3/flattened')
        net = tf.layers.dense (
            inputs=net,
            units=256,
            activation=tf.nn.relu,
            name='hidden_3/dense_256_relu'
        )

        return net

    return body_fn
