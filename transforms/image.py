#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from . import base

import skimage.transform
import numpy as np

class StateImageResizeTransform (base.IdentityTransform):
    def __init__ (self, environment, target_shape):
        super(StateImageResizeTransform, self).__init__ (
            environment=environment
        )

        self.state_shape = target_shape

    def copy (self):
        return type(self)(self.environment, self.state_shape)

    def state_transform (self, states):
        images = [skimage.transform.resize(image, self.state_shape, mode='constant') for image in states]
        return np.stack(images)

class StateLuminanceTransform (base.IdentityTransform):
    luminance_rgb = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)    

    def __init__ (self, environment):
        super(StateLuminanceTransform, self).__init__ (
            environment=environment
        )

        h, w, c = self.state_shape

        if c != 3:
            raise ValueError('Input image must have 3 channels, found {}.'.format(c))

        self.state_shape = (h, w, 1)

    def state_transform (self, states):
        states = np.dot(states, self.luminance_rgb)
        return states.reshape((-1,) + self.state_shape)

class StateImageStackTransform (base.IdentityTransform):
    def __init__ (self, environment, num_stacked_frames):
        super(StateImageStackTransform, self).__init__ (
            environment=environment
        )

        h, w, c = self.state_shape
        self.state_shape = (h, w, num_stacked_frames*c)

        self.num_stacked_frames = num_stacked_frames
        self.num_original_channels = c

    def copy (self):
        return type(self)(self.environment, self.num_stacked_frames)

    def reset_transform (self):
        self.stacked_frames = None

    def state_transform (self, states):
        if self.stacked_frames is None:
            self.stacked_frames = [np.zeros_like(states) for _ in xrange(self.num_stacked_frames)]

        self.stacked_frames.insert(0, states)
        self.stacked_frames.pop()

        return np.concatenate(self.stacked_frames, axis=-1)
