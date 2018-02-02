#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Policy (object):
    pass

class EpsilonGreedyPolicy (Policy):
    def __init__ (self, epsilon):
        self.epsilon = epsilon

    def __call__ (self, predictions):
        # random draw
        actions = np.random.choice (
            predictions.shape[-1],
            size=predictions.shape[0]
        )

        # draw per state if action is greedy or not
        random_draw = np.random.rand(actions.shape[0])
        greedy_actions = random_draw > self.epsilon

        # run predictions on greedy states
        if greedy_actions.any():
            actions[greedy_actions] = np.argmax(predictions[greedy_actions], axis=-1)

        return actions

    def get_epsilon (self):
        return self.epsilon

    def set_epsilon (self, epsilon):
        self.epsilon = epsilon
        return self.epsilon
