#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

DataConfig = collections.namedtuple (
    'DataConfig',
    [
        'dtype',
        'shape'
    ]
)
