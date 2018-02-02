#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['Modes', 'ModelSpecification']

class Modes (object):
    """Enum with model specification modes."""
    PREDICT = 0
    EVAL = 1
    TRAIN = 2

class ModelSpecification (object):
    """Full model specification."""
    def __init__ (
            self,
            mode=None, predictions=None,
            loss=None, train_op=None
    ):
        self.mode = mode
        self.predictions = predictions
        self.loss = loss
        self.train_op = train_op
