#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np

import agents
import configs
import environments
import policies
import runners
import memories
import models
import net_fns
import transforms

def make_dist_params (FLAGS):
    return dict (
        cluster = configs.distributed.get_cluster_local(FLAGS),
        task_index = FLAGS.task_index,
        is_chief = FLAGS.task_index == 0,
        config = configs.tf.session_configs['single_cpu']
    )

def make_local_params (FLAGS):
    return dict (
        cluster = None,
        task_index = 0,
        is_chief = True,
        config = None
    )

def main (FLAGS):
    # Net fn parameters
    hidden_units = [16]
    dropout_rate = 0.0

    # Model parameters
    model_dir = '/tmp/cartpole/model'
    learning_rate = 1e-3
    batch_size = 32

    # Distributed model parameters
    params_fn = make_local_params if FLAGS.task_index is None else make_dist_params
    dist_params = params_fn(FLAGS)

    # Policy parameters
    total_steps = 50*200

    epsilon_start = 1.0
    epsilon_stop = 0.00
    epsilon_steps = total_steps*0.95

    # Agent parameters
    gamma = 0.99

    environment = environments.GymEnvironment('Pong-v0')
    environment = transforms.image.StateImageResizeTransform(environment, (84, 84, 3))
    environment = transforms.image.StateLuminanceTransform(environment)
    environment = transforms.image.StateImageStackTransform(environment, num_stacked_frames=4)
    environment = environments.BatchEnvironment(environment, batch_size=batch_size)

    net_params = [
        # First conv layer
        dict (
            type='conv2d', filters=16, kernel_size=(9, 9), strides=(4, 4), activation='relu',
            name='hidden_0/conv_16_9x9_4x4_relu'
        ),

        # Second conv layer
        dict (
            type='conv2d', filters=32, kernel_size=(7, 7), strides=(2, 2), activation='relu',
            name='hidden_1/conv_32_7x7_2x2_relu'
        ),

        # Third conv layer
        dict (
            type='conv2d', filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu',
            name='hidden_2/conv_64_5x5_2x2_relu'
        ),

        # Fully connected layer
        dict(type='flatten', name='hidden_3/flattened'),
        dict(type='dense', units=128, activation='relu', name='hidden_3/dense_256_relu')
    ]

    model = models.Q.Model (
        # DataConfigs
        input_config=configs.DataConfig(dtype=np.float32, shape=environment.state_shape),
        output_config=configs.DataConfig(dtype=np.float32, shape=environment.action_shape),
        action_config=configs.DataConfig(dtype=np.int32, shape=()),
        target_config=configs.DataConfig(dtype=np.float32, shape=()),

        # Q.Model inputs
        net_fn=net_fns.make_net_fn(net_params=net_params),
        loss_type='mean_squared_error',

        # TF model inputs
        params=dict (
            # Tensorflow Model
            model_dir=model_dir,

            # Tensorflow Optimizer
            optimizer='Adam',
            learning_rate=learning_rate,

            # Tensorflow Distributed
            cluster=dist_params['cluster'],
            task_index=dist_params['task_index'],
            is_chief=dist_params['is_chief'],
            config=dist_params['config']
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

            special_episode = dist_params['is_chief'] and ((1+episodes) % 10 == 0)

            a = (epsilon_stop - epsilon_start)/epsilon_steps
            b = epsilon_start

            agent.policy.epsilon = a*step + b
            if agent.policy.epsilon < epsilon_stop:
                agent.policy.epsilon = epsilon_stop

            sub_episodes, rewards = runner.run_episode(render=special_episode, render_delay=None)
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
    parser = configs.distributed.add_argparse_args(parser)

    parser.add_argument (
        '--task_index', type=int, default=None,
        help='Index of worker in cluster.'
    )

    FLAGS = parser.parse_args()
    exit(main(FLAGS))
