#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

HOSTNAME = 'localhost'

# Port scheme used(?):
# Parameter server tasks: 2220-
# Worker tasks: 2330-
PS_PORT = 2220
WORKER_PORT = 2330

empty_cluster = {
    # Parameter servers
    'ps' : [
    ],

    # Training workers
    'worker' : [
    ],
}

def add_argparse_args (parser):
    parser.add_argument (
        '--ps_tasks', type=int, default=1,
        help='Number of parameter servers to launch.'
    )

    parser.add_argument (
        '--worker_tasks', type=int, default=1,
        help='Number of worker tasks to launch.'
    )

    return parser

def get_cluster (FLAGS):
    cluster = dict(empty_cluster)

    for index in xrange(FLAGS.ps_tasks):
        job = '{hostname}:{port_number}'.format (
            hostname=HOSTNAME,
            port_number=PS_PORT + index
        )

        cluster['ps'].append(job)

    for index in xrange(FLAGS.worker_tasks):
        job = '{hostname}:{port_number}'.format (
            hostname=HOSTNAME,
            port_number=WORKER_PORT + index
        )

        cluster['worker'].append(job)

    return cluster

def get_cluster_local (FLAGS):
    full_cluster = get_cluster(FLAGS)
    task_index = FLAGS.task_index

    return {
        'ps' : full_cluster['ps'],
        'worker' : { task_index : full_cluster['worker'][task_index] }
    }
