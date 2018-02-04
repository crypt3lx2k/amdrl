#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO:
# Should transforms copy environments or just take ownership?
import environments

class Transform (environments.Environment):
    def __init__ (self, environment):
        super(Transform, self).__init__ (
            state_shape=environment.state_shape,
            action_shape=environment.action_shape
        )

        self.environment = environment.copy()

    def copy (self):
        return type(self)(self.environment)

    def reset (self):
        return self.state_transform (
            self.environment.reset()
        )

    def step (self, actions):
        actions = self.action_transform(actions)
        states, rewards, terminals = self.environment.step(actions)

        return (
            self.state_transform(states),
            self.reward_transform(rewards),
            self.terminal_transform(terminals)
        )

    def render (self, close=False):
        return self.environment.render(close=close)

    def action_transform (self, actions):
        raise NotImplementedError()

    def state_transform (self, states):
        raise NotImplementedError()

    def reward_transform (self, rewards):
        raise NotImplementedError()

    def terminal_transform (self, terminals):
        raise NotImplementedError()

class IdentityTransform (Transform):
    def action_transform (self, actions):
        return actions

    def state_transform (self, states):
        return states

    def reward_transform (self, rewards):
        return rewards

    def terminal_transform (self, terminals):
        return terminals

class StateReshapeTransform (IdentityTransform):
    def __init__ (self, environment, target_shape):
        super(ReshapeTransform, self).__init__ (
            environment=environment
        )

        self.state_shape = target_shape

    def copy (self):
        return type(self)(self.environment, self.state_shape)

    def state_transform (self, states):
        return states.reshape((-1,) + self.state_shape)
