#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

class Runner (object):
    def __init__ (self, agent, environment):
        self.agent = agent
        self.environment = environment

        self.episodes = 0

    def run_episode (self, render=False, render_delay=None):
        episode_rewards = []

        states = self.environment.reset()
        self.agent.reset()

        sub_episodes = 0
        terminal = np.zeros(states.shape[0], dtype=np.bool)
        while not terminal.all():
            actions = self.agent.act(states)
            states, rewards, terminals = self.environment.step(actions)

            terminal |= terminals
            sub_episodes += sum(terminals)

            loss = self.agent.perceive(rewards, terminals)

            if render:
                self.environment.render()

                if render_delay is not None:
                    time.sleep(render_delay)

            episode_rewards.append(rewards)

        if render:
            self.environment.render(close=True)

        self.episodes += 1
        return sub_episodes, np.array(episode_rewards)
