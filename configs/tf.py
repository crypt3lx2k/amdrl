#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

session_configs = {
    'default' : None,

    'single_cpu' : dict (
        device_count = {
            'CPU' : 1,
            'GPU' : 0
        },
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
}
