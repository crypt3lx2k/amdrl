#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import layer_fns

def make_net_fn (net_params):
    """Makes a net function based on full specification."""

    layers = []
    for params in net_params:
        layer_fn = layer_fns.make_layer_fn(params)
        layers.append(layer_fn)

    def net_fn (features, training=False):
        """Builds full net."""
        net = features

        for i, layer in enumerate(layers):
            net = layer(net, training=training)

        return net

    return net_fn
