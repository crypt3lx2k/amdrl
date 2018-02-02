#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Memory (object):
    pass

class OneStepMemory (object):
    def __init__ (self):
        self.reset()

    def reset (self):
        # Previous step
        self.states = None
        self.actions = None
        self.rewards = None
        self.terminals = None

        # Current step
        self.states_next = None
        self.actions_next = None
        self.rewards_next = None
        self.terminals_next = None

        self.predictions = None
        self.eligible = None

    def act (self, states, actions, predictions):
        # Update one-step memory
        self.states = self.states_next
        self.states_next = states

        self.actions = self.actions_next
        self.actions_next = actions

        self.predictions = predictions

    def perceive (self, rewards, terminals):
        # Update one-step memory
        self.rewards = self.rewards_next
        self.rewards_next = rewards

        self.terminals = self.terminals_next
        self.terminals_next = terminals

    def has_batch (self):
        if self.states is None:
            return False

        return True

    def get_batch (self):
        eligible = self.eligible

        # If we're at the first state set all to eligible
        if eligible is None:
            eligible = np.ones_like(self.terminals, dtype=np.bool)

        # Keep track of eligible states
        if self.terminals is not None:
            self.eligible = np.logical_not(self.terminals)

        return map (
            lambda x : x[eligible],
            (
                self.states,
                self.actions,
                self.rewards,
                self.terminals,
                self.predictions,
                self.actions_next
            )
        )
