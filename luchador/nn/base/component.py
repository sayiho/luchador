"""Define GraphComponent class"""
from __future__ import absolute_import


class Node(object):  # pylint: disable=too-few-public-methods
    """Make subclass retrievable with get_node method"""
    def get_output_tensors(self):
        return []

    def get_update_operations(self):
        return None

    def get_parameters_to_train(self):
        return []

    def get_parameters_to_serialize(self):
        return []

    def get_parameter_variables(self, name=None):
        return []
