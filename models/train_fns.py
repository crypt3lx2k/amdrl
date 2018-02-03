#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

optimizer_fns = {
    'Adadelta'                : tf.train.AdadeltaOptimizer,
    'Adagrad'                 : tf.train.AdagradOptimizer,
    'Adam'                    : tf.train.AdamOptimizer,
    'Ftrl'                    : tf.train.FtrlOptimizer,
    'GradientDescent'         : tf.train.GradientDescentOptimizer,
    'Momentum'                : tf.train.MomentumOptimizer,
    'ProximalAdagrad'         : tf.train.ProximalAdagradOptimizer,
    'ProximalGradientDescent' : tf.train.ProximalGradientDescentOptimizer,
    'RMSProp'                 : tf.train.RMSPropOptimizer,
}

optimizer_defaults = {
    tf.train.AdadeltaOptimizer : dict (
        learning_rate=0.001,
        rho=0.95,
        epsilon=1e-08,
        use_locking=False,
    ),

    tf.train.AdagradOptimizer : dict (
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        use_locking=False,
    ),

    tf.train.AdamOptimizer : dict (
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
    ),

    tf.train.FtrlOptimizer : dict (
        learning_rate=0.001,
        learning_rate_power=-0.5,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0,
        l2_shrinkage_regularization_strength=0.0,
        use_locking=False,
    ),

    tf.train.GradientDescentOptimizer : dict (
        learning_rate=0.001,
        use_locking=False,
    ),

    tf.train.MomentumOptimizer : dict (
        learning_rate=0.001,
        momentum=0.9,
        use_nesterov=False,
        use_locking=False,
    ),

    tf.train.ProximalAdagradOptimizer : dict (
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0,
        use_locking=False,
    ),

    tf.train.ProximalGradientDescentOptimizer : dict (
        learning_rate=0.001,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0,
        use_locking=False,
    ),

    tf.train.RMSPropOptimizer : dict (
        learning_rate=0.001,
        decay=0.9,
        momentum=0.0,
        epsilon=1e-10,
        centered=False,
        use_locking=False,
    ),
}

def get_optimizer_fn (params):
    """Returns optimizer_fn and optimizer_params based on name."""

    # Lookup function and default parameters
    optimizer_name = params['optimizer']
    optimizer_fn = optimizer_fns[optimizer_name]
    optimizer_params = optimizer_defaults[optimizer_fn]

    # Override defaults if wanted
    for key in optimizer_params:
        if key in params:
            optimizer_params[key] = params[key]

    return optimizer_fn, optimizer_params

def make_train_fn (optimizer_fn, optimizer_params):
    """Makes a training function."""

    def train_fn (loss):
        """Builds a training operation using provided optimizer function."""
        # 
        global_step = tf.train.get_global_step()

        # Instantiate optimizer
        optimizer = optimizer_fn(**optimizer_params)

        # 
        grads_and_tvars = optimizer.compute_gradients(loss)

        # Get training operation
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients (
                grads_and_tvars,
                global_step=global_step
            )

        return train_op

    return train_fn
