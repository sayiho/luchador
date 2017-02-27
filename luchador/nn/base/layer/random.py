"""Module for implementing random source"""
from __future__ import absolute_import

from .base import BaseLayer


# pylint: disable=abstract-method
class BaseNormalNoise(BaseLayer):
    def __init__(self, mean=0.0, std=1.0, seed=None):
        super(BaseNormalNoise, self).__init__(mean=mean, std=std, seed=seed)
        self._rng = None


class BaseUniformNoise(BaseLayer):
    def __init__(self, low=0.0, high=1.0, seed=None):
        super(BaseUniformNoise, self).__init__(low=low, hight=high, seed=seed)
        self._rng = None
