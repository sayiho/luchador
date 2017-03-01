"""Define common interface for Cost classes"""
from __future__ import absolute_import

import abc
import logging

import luchador.util
from . import scope as scope_module

_LG = logging.getLogger(__name__)


class BaseCost(luchador.util.StoreMixin, object):
    """Define common interface for cost computation class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **args):
        """Validate args and set it as instance property

        See Also
        --------
        luchador.common.StoreMixin
            Underlying mechanism to store constructor arguments
        """
        super(BaseCost, self).__init__()
        self._store_args(**args)

        self.input = None
        self.output = None

    def __call__(self, target, prediction):
        """Convenience method to call `build`"""
        return self.build(target, prediction)

    def build(self, target, prediction):
        """Build cost between target and prediction

        Parameters
        ----------
        target : Tensor
            Tensor holding the correct prediction value.

        prediction : Tensor
            Tensor holding the current prediction value.

        Returns
        -------
        Tensor
            Tensor holding the cost between the given input tensors.
        """
        _LG.info(
            '  Building cost %s between target: %s and prediction: %s',
            type(self).__name__, target, prediction
        )
        with scope_module.variable_scope(self.args['name']):
            return self._build(target, prediction)

    @abc.abstractmethod
    def _build(self, target, prediction):
        pass


###############################################################################
# pylint: disable=abstract-method
class BaseSSE(BaseCost):
    """Compute Sum-Squared-Error for the given target and prediction

    .. math::
        loss = (target - prediction) ^ {2}

    Parameters
    ----------
    elementwise : Bool
        When True, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is reduced to
        scalar shape by taking average over batch and sum over feature.
        Defalut: False.
    """
    def __init__(self, elementwise=False, name='SSE'):
        super(BaseSSE, self).__init__(elementwise=elementwise, name=name)


class BaseSigmoidCrossEntropy(BaseCost):
    """Directory computes classification entropy from logit

    .. math::
        loss = \\frac{-1}{n} \\sum\\limits_{n=1}^N \\left[ p_n \\log
                \\hat{p}_n + (1 - p_n) \\log(1 - \\hat{p}_n) \\right]

    Parameters
    ----------
    elementwise : Bool
        When True, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is reduced to
        scalar shape by taking average over batch and sum over feature.
        Defalut: False.
    """
    def __init__(self, elementwise=False, name='SigmoidCrossEntropy'):
        super(BaseSigmoidCrossEntropy, self).__init__(
            elementwise=elementwise, name=name)
