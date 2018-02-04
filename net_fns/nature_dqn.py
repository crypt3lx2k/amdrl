#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

net_params = [
    # First conv layer
    dict (
        type='conv2d', filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu',
        name='hidden_0/conv_32_8x8_4x4_relu'
    ),

    # Second conv layer
    dict (
        type='conv2d', filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu',
        name='hidden_1/conv_64_4x4_2x2_relu'
    ),

    # Third conv layer
    dict (
        type='conv2d', filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
        name='hidden_2/conv_64_3x3_1x1_relu'
    ),

    # Fully connected layer
    dict(type='flatten'),
    dict(type='dense', units=256, activation='relu', name='hidden_3/dense_256_relu')
]
