#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

layer_fns = {
    'average_pooling1d'   : tf.layers.average_pooling1d,
    'average_pooling2d'   : tf.layers.average_pooling2d,
    'average_pooling3d'   : tf.layers.average_pooling3d,
    'batch_normalization' : tf.layers.batch_normalization,
    'conv1d'              : tf.layers.conv1d,
    'conv2d'              : tf.layers.conv2d,
    'conv2d_transpose'    : tf.layers.conv2d_transpose,
    'conv3d'              : tf.layers.conv3d,
    'conv3d_transpose'    : tf.layers.conv3d_transpose,
    'dense'               : tf.layers.dense,
    'dropout'             : tf.layers.dropout,
    'flatten'             : tf.layers.flatten,
    'max_pooling1d'       : tf.layers.max_pooling1d,
    'max_pooling2d'       : tf.layers.max_pooling2d,
    'max_pooling3d'       : tf.layers.max_pooling3d,
    'separable_conv2d'    : tf.layers.separable_conv2d,
}

layer_fn_has_training_switch = {
    tf.layers.batch_normalization : True,
    tf.layers.dropout             : True
}

activation_fns = {
    'sigmoid'  : tf.sigmoid,
    'tanh'     : tf.tanh,
    'relu'     : tf.nn.relu,
    'relu6'    : tf.nn.relu6,
    'crelu'    : tf.nn.crelu,
    'elu'      : tf.nn.elu,
    'selu'     : tf.nn.selu,
    'softplus' : tf.nn.softplus,
    'softsign' : tf.nn.softsign,
}

def make_layer_fn (params):
    """Makes layer function based on."""
    # Extract layer type
    layer_type = params.pop('type')

    # Get layer base function
    base_fn = layer_fns[layer_type]
    has_training_switch = layer_fn_has_training_switch.get (
        base_fn, False
    )

    # Get activation function if available
    if params.has_key('activation'):
        activation_name = params['activation']
        activation_fn = activation_fns[activation_name]
        params['activation'] = activation_fn

    def layer_fn (inputs, training=False):
        if has_training_switch:
            params['training'] = training

        return base_fn(inputs=inputs, **params)

    return layer_fn
