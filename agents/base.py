#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Agent (object):
    def __init__ (self, model, policy, memory):
        self.model = model
        self.policy = policy
        self.memory = memory

    def reset (self):
        self.memory.reset()

    def act (self, states):
        predictions = self.model.predict({'states' : states})
        actions = self.policy(predictions)

        self.memory.act(states, actions, predictions)
        return actions

    def perceive (self, rewards, terminals):
        self.memory.perceive(rewards, terminals)
        self.update()

    def update (self):
        pass
