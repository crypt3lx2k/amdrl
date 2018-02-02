#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import gym
import numpy as np

class Environment (object):
    """Base class for environments."""
    def __init__ (self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape

    def copy (self):
        raise NotImplementedError()

    def reset (self):
        raise NotImplementedError()

    def step (self, actions):
        raise NotImplementedError()

    def render (self):
        raise NotImplementedError()

class GymEnvironment (Environment):
    """Wrapper for OpenAI gym environments."""
    def __init__ (self, gym_id):
        environment = gym.make(gym_id)

        super(GymEnvironment, self).__init__ (
            state_shape=environment.observation_space.shape,
            action_shape=(environment.action_space.n,)
        )

        self.gym_id = gym_id
        self.environment = environment

    def wrap (self, state):
        return np.stack([state])

    def copy (self):
        return type(self)(self.gym_id)

    def reset (self):
        return self.wrap(self.environment.reset())

    def step (self, action):
        state, reward, terminal, info = self.environment.step(action[0])
        return map(self.wrap, (state, reward, terminal))

    def render (self, close=False):
        return self.environment.render(close=close)

class BatchEnvironment (Environment):
    """Batching class that presents multiple identical environments as one."""
    def __init__ (self, environment, batch_size):
        super(BatchEnvironment, self).__init__ (
            state_shape=environment.state_shape,
            action_shape=environment.action_shape
        )

        environments = [
            environment.copy() for _ in xrange(batch_size)
        ]

        self.environments = environments
        self.batch_size = batch_size

    def batch (self, states):
        return np.concatenate(states)

    def copy (self):
        return type(self)(self.environments[0], self.batch_size)

    def reset (self, terminals=None):
        return self.batch (
            map(lambda e : e.reset(), self.environments)
        )

    def step (self, actions):
        states = []
        rewards = []
        terminals = []

        for environment, action in zip(self.environments, actions):
            state, reward, terminal = environment.step([action])

            # Spin enviroment back up if it terminated
            if terminal:
                state = environment.reset()

            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)

        return map(self.batch, (states, rewards, terminals))

    def render (self, close=False):
        self.environments[0].render(close=close)
