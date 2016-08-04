from __future__ import absolute_import

import tensorflow as tf
from tensorflow import (  # nopep8
    Session as _Session,
    get_default_session,
)

from ..base import Session as BaseSession
from .tensor import Tensor, Operation

__all__ = ['Session']


def _get_full_class(cls):
    return '{}.{}'.format(cls.__module__, cls.__name__)

_TENSOR_CLASS_STR = _get_full_class(Tensor)
_OP_CLASS_STR = _get_full_class(Operation)


def _is_iteratable(l):
    try:
        list(l)
        return True
    except Exception:
        return False


def _parse_outputs(outputs):
    outputs_ = []
    if outputs is None:
        return outputs_

    if not _is_iteratable(outputs):
        outputs = [outputs]

    for output in outputs:
        if not isinstance(output, Tensor):
            raise ValueError(
                '`outputs` must be [list of] {}. Given: {}'
                .format(_TENSOR_CLASS_STR, _get_full_class(type(output))))
        outputs_.append(output.tensor)
    return outputs_


def _parse_updates(updates):
    ret = []
    if not updates:
        return ret

    if not _is_iteratable(updates):
        updates = [updates]

    for update in updates:
        if not isinstance(update, Operation):
            raise ValueError(
                '`updates` must be [list of] {}. Given: {}'
                .format(_OP_CLASS_STR, _get_full_class(type(update))))
        ret.append(update.op)
    return ret


def _construct_fetches(outputs, updates):
    return _parse_outputs(outputs) + _parse_updates(updates)


def _construct_feed_dict(inputs, givens):
    feed_dict = {}
    if not inputs:
        pass
    elif isinstance(inputs, dict):
        for key, value in inputs.items():
            feed_dict[key.tensor] = value
    elif isinstance(inputs, list):
        for key, value in inputs:
            feed_dict[key.tensor] = value
    else:
        raise ValueError(
            '`inputs` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(inputs)))

    if not givens:
        pass
    elif isinstance(givens, dict):
        for key, value in givens.items():
            feed_dict[key.tensor] = value
    elif isinstance(givens, list):
        for key, value in givens:
            feed_dict[key.tensor] = value
    else:
        raise ValueError(
            '`givens` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(givens)))
    return feed_dict


class Session(BaseSession):
    def __init__(self, graph=None, config=None, **kwargs):
        self.session = _Session('', graph, config)

    @property
    def graph(self):
        return self.session.graph

    def run(self, name, outputs=[], inputs={}, updates=None, givens=None):
        """

        Args:
          name (str): Not used. Compatibility for theano backend
          outputs (list of Tensors):
          inputs (dict):
          updates (Operation or list of Operations)
          givens (dict):
        """
        fetches = _construct_fetches(outputs, updates)
        feed_dict = _construct_feed_dict(inputs, givens)
        values = self.session.run(fetches, feed_dict=feed_dict)
        if _is_iteratable(outputs):
            return values[:len(outputs)]
        return values[0]

    def close(self):
        return self.session.close()

    def initialize(self):
        self.session.run(tf.initialize_all_variables())