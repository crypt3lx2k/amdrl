#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base

import numpy as np

__all__ = [
    'LearningAgent',
    'SarsaAgent'
]

class QAgent (base.Agent):
    def __init__ (self, model, policy, memory, gamma):
        super(QAgent, self).__init__(model=model, policy=policy, memory=memory)

        self.gamma = gamma

    def update (self, states, targets, actions):
        loss = self.model.train (
            {'states'  : states},
            {'targets' : targets,
             'actions' : actions}
        )

        # print (
        #     'loss={:04.2f} eps={:04.2f} '
        #     'mean={:04.2f} std={:04.2f} '
        #     'batch_size={} id={}'.format (
        #         loss,
        #         self.policy.epsilon,
        #         targets.mean(),
        #         targets.std(),
        #         len(states),
        #         self.model.server.target
        #     )
        # )

        return loss

class LearningAgent (QAgent):
    def update (self):
        if not self.memory.has_batch():
            return None

        (
            states, actions, rewards,
            terminals, predictions, _
        ) = self.memory.get_batch()

        # Q-learning update
        targets = predictions.max(axis=-1)

        targets[terminals] = 0.0
        targets = self.gamma*targets + rewards

        return super(LearningAgent, self).update(states, targets, actions)

class SarsaAgent (QAgent):
    def update (self):
        if not self.memory.has_batch():
            return None

        (
            states, actions, rewards,
            terminals, predictions, next_actions
        ) = self.memory.get_batch()

        # Sarsa update
        targets = predictions[np.indices(next_actions.shape), next_actions].flatten()

        targets[terminals] = 0.0
        targets = self.gamma*targets + rewards

        return super(SarsaAgent, self).update(states, targets, actions)
