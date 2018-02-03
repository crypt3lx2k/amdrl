#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from . import distributed
from . import tf

__all__ = ['DataConfig', 'distributed', 'tf']

DataConfig = collections.namedtuple (
    'DataConfig',
    [
        'dtype',
        'shape'
    ]
)
