"""Implement Layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging

import theano.tensor as T

from luchador.nn.base import (
    layer as base_layer,
    initializer as base_initializer,
)
from . import scope, wrapper, initializer

__all__ = [
    'LayerMixin',
    'Dense', 'Conv2D',
    'ReLU', 'Softplus',
    'Sigmoid', 'Softmax',
    'Tanh', 'Sin', 'Cos',
    'Flatten', 'TrueDiv', 'Mean',
    'Concat', 'Add', 'Sub',
    'BatchNormalization',
    'NHWC2NCHW', 'NCHW2NHWC',
]

_LG = logging.getLogger(__name__)


class LayerMixin(object):  # pylint: disable=too-few-public-methods
    """Implement the following common Layer methods in Theano

    - ``_get_update_operation``

    """
    def _get_update_operation(self):
        return wrapper.Operation(self.update_operations)


def _get_initializers(cfg, with_bias):
    """Initializer for Dense and Conv2D"""
    w_cfg = cfg.get('weight')
    ret = {}
    ret['weight'] = (
        base_initializer.get_initializer(w_cfg['typename'])(**w_cfg['args'])
        if w_cfg else initializer.Xavier()
    )

    if with_bias:
        _cfg = cfg.get('bias')
        ret['bias'] = (
            base_initializer.get_initializer(_cfg['typename'])(**_cfg['args'])
            if _cfg else initializer.Constant(0.1)
        )
    return ret


class Dense(LayerMixin, base_layer.BaseDense):
    """Implement Dense layer in Theano.

    See :any:`BaseDense` for detail.
    """
    def _instantiate_parameters(self, n_inputs, dtype):
        initializers = _get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = (n_inputs, self.args['n_nodes'])
        w_init = initializers['weight']
        self._add_parameter('weight', scope.get_variable(
            name='weight', shape=w_shape, initializer=w_init, dtype=dtype))

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            b_init = initializers['bias']
            self._add_parameter('bias', scope.get_variable(
                name='bias', shape=b_shape, initializer=b_init, dtype=dtype))

    def _build(self, input_tensor):
        input_shape = input_tensor.shape

        if not len(input_shape) == 2:
            raise ValueError('Input tensor must be 2D. '
                             'Insted of {}'.format(len(input_shape)))

        if not self.parameter_variables:
            self._instantiate_parameters(input_shape[1], input_tensor.dtype)

        weight = self._get_parameter('weight').unwrap()
        output = T.dot(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = output + bias
        output_shape = (input_shape[0], self.args['n_nodes'])
        return wrapper.Tensor(output, shape=output_shape, name='output')


def _map_border_mode(padding):
    if isinstance(padding, str):
        mode = padding.lower()
        return 'half' if mode == 'same' else mode
    return padding


def _is_int_list(list_, length=2):
    return len(list_) == length and all([isinstance(e, int) for e in list_])


def _validate_padding(padding):
    msg = ('`padding` must be either str ("valid", "full", "half" or '
           '"same"), int or tuple of two int')

    if isinstance(padding, int):
        return

    if isinstance(padding, str):
        if padding.lower() in ['full', 'half', 'same', 'valid']:
            return
        raise ValueError(msg)

    try:
        if _is_int_list(padding, length=2):
            return
    except TypeError:
        pass

    raise ValueError(msg)


def _validate_strides(strides):
    if isinstance(strides, int):
        return
    try:
        if _is_int_list(strides, length=2):
            return
    except TypeError:
        pass

    raise ValueError('`strides` must be either int or tuple of two int')


class Conv2D(LayerMixin, base_layer.BaseConv2D):
    """Implement Conv2D layer in Theano.

    See :any:`BaseConv2D` for detail.
    """
    def _validate_args(self, padding, strides, **_):
        _validate_padding(padding)
        _validate_strides(strides)

    ###########################################################################
    def _instantiate_parameters(self, n_inputs, dtype):
        initializers = _get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = (self.args['n_filters'], n_inputs,
                   self.args['filter_height'], self.args['filter_width'])
        w_init = initializers['weight']
        self._add_parameter('weight', scope.get_variable(
            name='weight', shape=w_shape, initializer=w_init, dtype=dtype))

        if self.args['with_bias']:
            b_shape = (self.args['n_filters'],)
            b_init = initializers['bias']
            self._add_parameter('bias', scope.get_variable(
                name='bias', shape=b_shape, initializer=b_init, dtype=dtype))

    def _get_subsample(self):
        if isinstance(self.args['strides'], int):
            return (self.args['strides'], self.args['strides'])
        return self.args['strides']

    def _get_border_mode(self):
        return _map_border_mode(self.args['padding'])

    def _get_output_shape(self, input_shape, filter_shape):
        """Compute output shape

        Parameters
        ----------
        input_shape : tuple
            Input shape in order of (batch, n_input_channels, row, col)

        filter_shape : tuple
            Filter shape in order of (n_filters, n_input_channels, rows, cols)
        """
        # TODO: Add warning if
        # parts of image are not covered because of subsampling
        f_row, f_col = filter_shape[2:4]
        in_row, in_col = input_shape[2:4]
        sub_row, sub_col = self._get_subsample()
        border_mode = self._get_border_mode()
        # Process padding
        if border_mode in ['full', 'valid']:
            pass
        elif border_mode == 'half':
            in_row += 2 * (f_row // 2)
            in_col += 2 * (f_col // 2)
        elif isinstance(border_mode, int):
            in_row += 2 * border_mode
            in_col += 2 * border_mode
        else:
            in_row += 2 * border_mode[0]
            in_col += 2 * border_mode[1]
        # Process convolution
        if border_mode == 'full':
            out_row = (in_row + f_row - 2) // sub_row + 1
            out_col = (in_col + f_col - 2) // sub_col + 1
        else:
            out_row = (in_row - f_row) // sub_row + 1
            out_col = (in_col - f_col) // sub_col + 1
        # Reconstruct
        n_batches, n_filters = input_shape[0], filter_shape[0]
        output_shape = (n_batches, n_filters, out_row, out_col)
        return output_shape

    def _build(self, input_tensor):
        """Build 2D conolution operation of the input tensor

        Parameters
        ----------
        input_tensor : Tensor
            4D Tensor with shape (batch, #input channel, row, col)

        Returns
        -------
        Tensor
            4D Tensor with shape (batch, #output channel, row, col)
        """
        input_shape = input_tensor.shape
        _LG.debug('    input_shape: %s', input_shape)
        _LG.debug('    border_mode: %s', self._get_border_mode())

        if not len(input_shape) == 4:
            raise ValueError('Input tensor must be 4D. '
                             'Insted of {}'.format(len(input_shape)))

        if not self.parameter_variables:
            self._instantiate_parameters(input_shape[1], input_tensor.dtype)

        filters = self._get_parameter('weight').unwrap()
        filter_shape = filters.get_value().shape
        subsample = self._get_subsample()
        border_mode = self._get_border_mode()

        output_tensor = T.nnet.conv2d(
            input_tensor.unwrap(), filters=filters,
            input_shape=input_shape, filter_shape=filter_shape,
            border_mode=border_mode, subsample=subsample)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            bias = bias.dimshuffle(('x', 0, 'x', 'x'))
            output_tensor = bias + output_tensor

        output_shape = self._get_output_shape(input_shape, filter_shape)
        _LG.debug('    output_shape: %s', output_shape)
        return wrapper.Tensor(output_tensor, shape=output_shape, name='output')


class ReLU(LayerMixin, base_layer.BaseReLU):
    """Implement ReLU layer in Theano.

    See :any:`BaseReLU` for detail.
    """
    def _build(self, input_tensor):
        """Build rectified linear activation operation on input tensor"""
        input_shape = input_tensor.shape
        output_tensor = T.nnet.relu(input_tensor.unwrap())
        return wrapper.Tensor(output_tensor, shape=input_shape, name='output')


class Sigmoid(LayerMixin, base_layer.BaseSigmoid):
    """Implement Sigmoid layer in Theano.

    See :any:`BaseSigmoid` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.sigmoid(input_tensor.unwrap())
        return wrapper.Tensor(output_tensor, shape=input_shape, name='output')


class Tanh(LayerMixin, base_layer.BaseTanh):
    """Implement Tanh layer in Theano.

    See :any:`BaseTanh` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.tanh(input_tensor.unwrap())
        return wrapper.Tensor(output_tensor, shape=input_shape, name='output')


class Sin(LayerMixin, base_layer.BaseSin):
    """Implement Sin layer in Theano

    See :any:`BaseSin` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.sin(input_tensor.unwrap())
        return wrapper.Tensor(output_tensor, shape=input_shape, name='output')


class Cos(LayerMixin, base_layer.BaseCos):
    """Implement Cos layer in Theano

    See :any:`BaseSin` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.cos(input_tensor.unwrap())
        return wrapper.Tensor(output_tensor, shape=input_shape, name='output')


class Softmax(LayerMixin, base_layer.BaseSoftmax):
    """Implement Softmax layer in Theano.

    See :any:`BaseSoftmax` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.softmax(input_tensor.unwrap())
        return wrapper.Tensor(output_tensor, shape=input_shape, name='output')


class Softplus(LayerMixin, base_layer.BaseSoftplus):
    """Implemente Softplus layer in Theano.

    See :any:`BaseSoftplus` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.softplus(input_tensor.unwrap())
        return wrapper.Tensor(output_tensor, shape=input_shape, name='output')


###############################################################################
class Flatten(LayerMixin, base_layer.BaseFlatten):
    """Implement Flatten layer in Theano

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        n_nodes = int(reduce(lambda r, d: r*d, input_shape[1:], 1))

        _LG.debug('    Input shape: %s', input_shape)
        _LG.debug('    #Nodes     : %s', n_nodes)

        output_shape = (input_shape[0] or -1, n_nodes)
        output_tensor = T.reshape(input_tensor.unwrap(), output_shape)
        _LG.debug('    output_shape: %s', output_shape)
        return wrapper.Tensor(output_tensor, shape=output_shape, name='output')


def _compute_concat_shape(shapes, axis):
    _shape = [None] * len(shapes[0])
    _shape[axis] = 0
    for shape in shapes:
        for i, val in enumerate(shape):
            if i == axis:
                if _shape[i] is None or val is None:
                    _shape[i] = None
                else:
                    _shape[i] += val
            else:
                if _shape[i] is None or val is None:
                    _shape[i] = _shape[i] or val
                else:
                    if not _shape[i] == val:
                        raise ValueError('Inconsistent shape')
    return _shape


class Concat(LayerMixin, base_layer.BaseConcat):
    """Implement Concat layer in Theano

    See :any: `BaseConcat` for detail.
    """
    def _build(self, var_list):
        if len(var_list) < 2:
            raise ValueError('var_list must contain more than 1 tensor')
        axis = self.args['axis']

        tensor_list = [var.unwrap() for var in var_list]
        shape_list = [var.shape for var in var_list]
        shape = _compute_concat_shape(shape_list, axis)
        output = T.concatenate(tensor_list=tensor_list, axis=axis)
        return wrapper.Tensor(output, shape=shape, name='output')


class Add(LayerMixin, base_layer.BaseAdd):
    """Implement Add layer in Theano

    See :any: `BaseAdd` for detail.
    """
    def _build(self, var_list):
        if len(var_list) < 2:
            raise ValueError('var_list must contain more than 1 tensor')

        ret = var_list[0]
        for var in var_list[1:]:
            ret = ret + var
        return ret


class Sub(LayerMixin, base_layer.BaseAdd):
    """Implement Sub layer in Theano

    See :any: `BaseSub` for detail.
    """
    def _build(self, var_list):
        if len(var_list) == 2:
            raise ValueError('var_list must be 2 tensors')

        return var_list[0] - var_list[1]


###############################################################################
class TrueDiv(LayerMixin, base_layer.BaseTrueDiv):
    """Implement TrueDiv layer in Theano.

    See :any:`BaseTrueDiv` for detail.
    """
    def _instantiate_denominator(self, dtype):
        self.denom = T.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def _build(self, input_tensor):
        if self.denom is None:
            self._instantiate_denominator(input_tensor.dtype)
        output = input_tensor.unwrap() / self.denom
        return wrapper.Tensor(output, shape=input_tensor.shape, name='output')


class Mean(LayerMixin, base_layer.BaseMean):
    """Implement Mean layer in Theano.

    See :any:`BaseMean` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.mean(**self.args)


###############################################################################
class BatchNormalization(LayerMixin, base_layer.BaseBatchNormalization):
    """Implement BN layer in Theano.

    See :any:`BaseBatchNormalization` for detail.
    """
    def _instantiate_parameters(self, input_shape, dtype):
        dim = len(input_shape)
        shape = tuple(input_shape[i] for i in range(dim) if i == 1)
        self._axes = tuple(i for i in range(dim) if not i == 1)
        self._pattern = tuple((0 if i == 1 else 'x') for i in range(dim))

        _LG.debug('    Shape: %s', shape)
        _LG.debug('     Axes: %s', self._axes)
        _LG.debug('  Pattern: %s', self._pattern)

        mean = scope.get_variable(
            name='mean', shape=shape, trainable=False,
            initializer=initializer.Constant(0), dtype=dtype)
        var = scope.get_variable(
            name='var', shape=shape, trainable=False,
            initializer=initializer.Constant(1), dtype=dtype)

        scale = scope.get_variable(
            name='scale', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['scale']), dtype=dtype)
        offset = scope.get_variable(
            name='offset', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['offset']), dtype=dtype)

        self._add_parameter('mean', mean)
        self._add_parameter('var', var)
        self._add_parameter('scale', scale)
        self._add_parameter('offset', offset)

    def _build(self, input_tensor):
        if not self.parameter_variables:
            self._instantiate_parameters(
                input_tensor.shape, input_tensor.dtype)

        input_tensor_ = input_tensor.unwrap()

        mean_acc = self._get_parameter('mean').unwrap()
        var_acc = self._get_parameter('var').unwrap()
        scale = self._get_parameter('scale').unwrap()
        offset = self._get_parameter('offset').unwrap()

        if self.args['learn']:
            decay = self.args['decay']
            mean_in = input_tensor_.mean(axis=self._axes)
            var_in = input_tensor_.var(self._axes)

            new_mean_acc = decay * mean_acc + (1 - decay) * mean_in
            new_var_acc = decay * var_acc + (1 - decay) * var_in

            self._add_update(mean_acc, new_mean_acc)
            self._add_update(var_acc, new_var_acc)

            mean_acc = new_mean_acc
            var_acc = new_var_acc

        mean_acc = mean_acc.dimshuffle(self._pattern)
        var_acc = var_acc.dimshuffle(self._pattern)
        scale = scale.dimshuffle(self._pattern)
        offset = offset.dimshuffle(self._pattern)

        stdi = T.inv(T.sqrt(var_acc + self.args['epsilon']))
        output = scale * (input_tensor_ - mean_acc) * stdi + offset
        return wrapper.Tensor(output, shape=input_tensor.shape, name='output')


###############################################################################
class NHWC2NCHW(LayerMixin, base_layer.BaseNHWC2NCHW):
    """See :any:`BaseNHWC2NCHW` for detail."""
    def _build(self, input_tensor):
        output_tensor = input_tensor.unwrap().dimshuffle(0, 3, 1, 2)

        shape = input_tensor.shape
        output_shape = (shape[0], shape[3], shape[1], shape[2])
        return wrapper.Tensor(output_tensor, shape=output_shape, name='output')


class NCHW2NHWC(LayerMixin, base_layer.BaseNCHW2NHWC):
    """See :any:`BaseNCHW2NHWC` for detail."""
    def _build(self, input_tensor):
        output_tensor = input_tensor.unwrap().dimshuffle(0, 2, 3, 1)

        shape = input_tensor.shape
        output_shape = (shape[0], shape[2], shape[3], shape[1])
        return wrapper.Tensor(output_tensor, shape=output_shape, name='output')