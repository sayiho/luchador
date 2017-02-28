"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import abc


def _flatten(nested_list):
    ret = []
    for elem in nested_list:
        ret.extend(elem)
    return ret


class BaseModel(object):  # pylint: disable=too-few-public-methods
    """Base Model class"""
    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def _get_components(self):
        """Fetch internal components (model, layers) in list"""
        pass

    ###########################################################################
    # Functions for retrieving variables, tensors and operations
    def get_parameter_variables(self):
        """Get parameter Variables

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        return _flatten([
            comp.get_parameter_variables() for comp in self._get_components()
        ])

    def get_parameters_to_train(self):
        """Get parameter Variables to be fet to gradient computation.

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        return _flatten([
            comp.get_parameters_to_train() for comp in self._get_components()
        ])

    def get_parameters_to_serialize(self):
        """Get parameter Variables to be serialized.

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        return _flatten([
            cm.get_parameters_to_serialize() for cm in self._get_components()
        ])

    def get_output_tensors(self):
        """Get Tensor objects which represent the output of each layer

        Returns
        -------
        list
            List of Tensors each of which hold output from layer
        """
        return [comp.output for comp in self._get_components()]

    def get_update_operations(self):
        """Get update opretaions from each layer

        Returns
        -------
        list
            List of update operations from each layer
        """
        return _flatten([
            comp.get_update_operations() for comp in self._get_components()
        ])
