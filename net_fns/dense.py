#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['make_net_params']

def make_net_params (hidden_units, activation='tanh', dropout_rate=None):
    """Makes a densely connected net configuration."""

    net_params = []
    net_params.append(dict(type='flatten', name='inputs/flattened'))

    for i, units in enumerate(hidden_units):
        net_params.append (
            dict (
                type='dense', units=units, activation=activation,
                name='hidden_{}/dense_{}'.format(i, units)
            )
        )

        if dropout_rate is None or dropout_rate <= 0.0:
            continue

        net_params.append (
            dict(type='dropout', rate=dropout_rate, name='hidden_{}/dropout'.format(i))
        )

    return net_params
