#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import spec

import tensorflow as tf

def make_model_fn (inference_fn, loss_fn, train_fn):
    """Makes a full model function."""

    def model_fn (features, labels, mode, params):
        """Full model function for inference, eval and training."""
        # Training flag
        training = (mode == spec.Modes.TRAIN)

        # Build inference
        with tf.variable_scope('inference'):
            predictions = inference_fn (
                features=features,
                training=training
            )

        # Model specification
        specification = spec.ModelSpecification (
            mode=mode,
            predictions=predictions
        )

        # Return early specification if only used for inference
        if mode == spec.Modes.PREDICT:
            return specification

        # Build loss and add to specification
        with tf.variable_scope('loss'):
            loss = loss_fn(predictions, labels)

        specification.loss = loss

        # Early return for eval mode
        if mode == spec.Modes.EVAL:
            return specification

        # Build training operation add to specification then return
        with tf.variable_scope('train'):
            train_op = train_fn(loss)

        specification.train_op = train_op
        return specification

    return model_fn
