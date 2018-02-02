#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np

import agents
import body_fns
import environments
import policies
import runners
import memories
import models

import configs_distributed
import configs_tf
from configs import DataConfig

def main (FLAGS):
    # Body fn parameters
    hidden_units = [16]
    dropout_rate = 0.0

    # Model parameters
    model_dir = '/tmp/cartpole/model'
    learning_rate = 1e-3
    batch_size = 16

    # Distributed model parameters
    cluster = None
    task_index = 0
    is_chief = True
    session_config = None

    if FLAGS.task_index is not None:
        cluster = configs_distributed.get_cluster_local(FLAGS)
        task_index = FLAGS.task_index
        is_chief = task_index == 0
        session_config = configs_tf.session_configs['single_cpu']

    # Policy parameters
    epsilon_start = 1.0
    epsilon_stop = 0.05
    epsilon_steps = 100000//2

    total_steps = 2*epsilon_steps

    # Agent parameters
    gamma = 0.99

    base_environment = environments.GymEnvironment('CartPole-v0')
    environment = environments.BatchEnvironment (
        environment=base_environment,
        batch_size=batch_size
    )

    model = models.Q.Model (
        input_config=DataConfig(dtype=np.float32, shape=environment.state_shape),
        output_config=DataConfig(dtype=np.float32, shape=environment.action_shape),
        action_config=DataConfig(dtype=np.int32, shape=()),
        target_config=DataConfig(dtype=np.float32, shape=()),
        body_fn=body_fns.dense.make_body_fn(hidden_units, dropout_rate),
        loss_type='mean_squared_error',
        params=dict (
            # Tensorflow Model
            model_dir=model_dir,
            
            # Tensorflow Optimizer
            optimizer='GradientDescent',
            learning_rate=learning_rate,

            # Tensorflow Distributed
            cluster=cluster,
            task_index=task_index,
            is_chief=is_chief,
            config=session_config
        )
    )

    policy = policies.EpsilonGreedyPolicy(epsilon=epsilon_start)
    memory = memories.OneStepMemory()
    agent = agents.Q.SarsaAgent(model=model, policy=policy, memory=memory, gamma=gamma)
    runner = runners.Runner(agent=agent, environment=environment)

    with model:
        start_step = model.step()
        step = start_step
        episodes = 0

        while step - start_step < total_steps:
            if model.session.should_stop():
                break

            special_episode = is_chief and ((1+episodes) % 10 == 0)

            a = (epsilon_stop - epsilon_start)/epsilon_steps
            b = epsilon_start

            agent.policy.epsilon = a*step + b
            if agent.policy.epsilon < epsilon_stop:
                agent.policy.epsilon = epsilon_stop

            sub_episodes, rewards = runner.run_episode(render=special_episode, render_delay=1.0/60)
            rewards = rewards.mean(axis=-1)
            reward = rewards.sum()

            print('eps={:.2f} reward={:.2f} sub_episodes={} id={}'.format (
                agent.policy.epsilon, reward, sub_episodes, agent.model.server.target
            ))

            step = model.step()
            episodes += 1

    return 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser = configs_distributed.add_argparse_args(parser)

    parser.add_argument (
        '--task_index', type=int, default=None,
        help='Index of worker in cluster.'
    )

    FLAGS = parser.parse_args()
    exit(main(FLAGS))
