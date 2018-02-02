#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base

import numpy as np
import tensorflow as tf

__all__ = ['Model']

loss_terms = {
    'mean_squared_error' : tf.losses.mean_squared_error,
    'huber_loss' : tf.losses.huber_loss,
}

def make_input_fn (input_config, target_config, action_config):
    """Makes the input function for placeholders."""

    def input_fn ():
        """Input placeholder function."""
        features = {
            'states' : base.build_placeholder(input_config, name='states')
        }

        labels = {
            'targets' : base.build_placeholder(target_config, name='targets'),
            'actions' : base.build_placeholder(action_config, name='actions')
        }

        return features, labels

    return input_fn

def make_inference_fn (body_fn, output_config):
    """Makes full inference function based on body."""

    def inference_fn (features, training=False):
        """Feeds correct input to body function and connects head."""
        with tf.variable_scope('body'):
            # Feed inputs through body
            net = body_fn(features=features['states'], training=training)

            # Flatten body output
            net = tf.layers.flatten(net, name='flattened')

        # Output layer
        output_units = np.prod(output_config.shape)
        logits = tf.layers.dense (
            inputs=net,
            units=output_units,
            name='logits'
        )

        # Reshape if necessary
        logits_shape = logits.shape.as_list()[1:]
        logits_shape = tuple(logits_shape)

        if logits_shape != output_config.shape:
            logits = tf.reshape (
                tensor=logits,
                shape=(-1,) + output_config.shape,
                name='logits/reshaped'
            )

        return {'q_values' : logits}

    return inference_fn

def make_loss_fn (loss_term):
    """Makes a loss function based on input loss type."""

    def loss_fn (predictions, labels):
        """Q-value loss function."""
        # Get number of outputs
        predictions = predictions['q_values']
        output_classes = predictions.shape[-1]

        # One hot mask for action taken
        action_mask = tf.one_hot(labels['actions'], output_classes, name='action_mask')

        # Expand targets so they match shape of predictions
        targets = labels['targets']
        targets = tf.expand_dims(targets, axis=-1, name='targets/expanded')

        # Filter out targets based on action taken
        targets *= action_mask

        # Final loss between predictions and Q-value targets
        loss = loss_term (
            predictions=predictions, labels=targets, weights=action_mask
        )

        return loss

    return loss_fn

class Model (base.TFModel):
    """TensorFlow model for Q-value estimation."""

    loss_types = loss_terms.keys()

    def __init__ (
            self,
            input_config, output_config,
            action_config, target_config,
            body_fn, loss_type,
            params
    ):
        # Setup super-class
        super(Model, self).__init__ (
            input_config=input_config,
            output_config=output_config,
            params=params
        )

        # Q-value specific configs
        self.action_config = action_config
        self.target_config = target_config

        # Set up functions to pass to model function
        input_fn = make_input_fn(input_config, target_config, action_config)
        inference_fn = make_inference_fn(body_fn, output_config)
        loss_fn = make_loss_fn(loss_terms[loss_type])

        # Build graph
        self.build(input_fn=input_fn, inference_fn=inference_fn, loss_fn=loss_fn)

    def predict (self, features):
        """Returns Q-value prediction based on input features."""
        return super(Model, self).predict(features)['q_values']
