"""Module for implementing random source"""
from __future__ import absolute_import

import tensorflow as tf

from ...base import layer as base_layer
from ..wrapper import Tensor


class _NoiseMixin(object):  # pylint: disable=too-few-public-methods
    def _build(self, input_tensor):
        shape = input_tensor.shape
        dtype = input_tensor.dtype
        noise = self._sample(shape=shape, dtype=dtype)

        tensor = input_tensor.unwrap() + noise
        return Tensor(tensor=tensor, name='output')


class NormalNoise(_NoiseMixin, base_layer.BaseNormalNoise):
    """Implement NormalNoise in theano backend.

    See :any:`BaseNormalNoise` for the detail
    """
    def _sample(self, shape, dtype):
        mean, std = self.args['mean'], self.args['std']
        return tf.random_normal(
            shape=shape, mean=mean, stddev=std,
            dtype=dtype, seed=self.args['seed'],
        )


class UniformNoise(_NoiseMixin, base_layer.BaseUniformNoise):
    """Implement UniformNoise in theano backend.

    See :any:`BaseUniformNoise` for the detail
    """
    def _sample(self, shape, dtype):
        minval, maxval = self.args['low'], self.args['high']
        return tf.random_uniform(
            shape=shape, minval=minval, maxval=maxval,
            dtype=dtype, seed=self.args['seed']
        )
