"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = ['BasePool2D']

_LG = logging.getLogger(__name__)

# pylint: disable=abstract-method


class BasePool2D(BaseLayer):
    def __init__(
            self, kernel_height, kernel_width, strides, mode,
            padding='VALID'):
        super(BasePool2D, self).__init__(
            kernel_height=kernel_height, kernel_width=kernel_width,
            strides=strides, mode=mode, padding=padding)
