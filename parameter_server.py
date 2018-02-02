#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configs_distributed
import configs_tf

import tensorflow as tf

def main (FLAGS):
    cluster = configs_distributed.get_cluster(FLAGS)
    cluster = tf.train.ClusterSpec(cluster)

    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)

    server = tf.train.Server (
        cluster,
        job_name='ps',
        config=configs_tf.session_configs['single_cpu'],
        task_index=FLAGS.task_index
    )

    return server.join()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser = configs_distributed.add_argparse_args(parser)

    parser.add_argument (
        '--task_index', type=int, required=True,
        help='Index of parameter server in cluster.'
    )

    parser.add_argument (
        '--debug', action='store_true',
        help='Output debugging information.'
    )

    FLAGS = parser.parse_args()
    exit(main(FLAGS))
