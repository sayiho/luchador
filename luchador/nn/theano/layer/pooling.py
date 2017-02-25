"""Implement Layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging

from theano.tensor.signal.pool import pool_2d

from ...base import layer as base_layer
from .. import wrapper

__all__ = ['Pool2D']

_LG = logging.getLogger(__name__)


class Pool2D(base_layer.BasePool2D):
    """Implement Pool2D layer in Theano.

    See :any:`BasePool2D` for detail.
    """
    def _build(self, input_tensor):
        ws = (self.args['kernel_height'], self.args['kernel_width'])
        stride, mode = self.args['strides'], self.args['mode']
        # TODO: use padding and compute output shape correctly
        _tensor = pool_2d(
            input_tensor.unwrap(), ws=ws, stride=stride, mode=mode, pad=(0, 0), ignore_border=True)

        input_shape = input_tensor.shape
        new_height = input_tensor.shape[2] // self.args['kernel_height']
        new_width = input_tensor.shape[3] // self.args['kernel_width']

        output_shape = (input_shape[0], input_shape[1], new_height, new_width)

        return wrapper.Tensor(_tensor, shape=output_shape, name='output')
