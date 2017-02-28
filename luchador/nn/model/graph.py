from __future__ import absolute_import

from .base_model import BaseModel


class Graph(BaseModel):
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def get_parameter_variables(self):
        """Get parameter Variables

        Returns
        -------
        list
            List of Variables from interanal models.
        """
        ret = []
        for node in self.nodes:
            ret.extend(node.get_parameter_variables())
        return ret

    def get_parameters_to_train(self):
        """Get parameter Variables to be fet to gradient computation.

        Returns
        -------
        list
            List of Variables from interanal models.
        """
        ret = []
        for node in self.nodes:
            ret.extend(node.get_parameters_to_train())
        return ret

    def get_parameters_to_serialize(self):
        """Get parameter Variables to be serialized.

        Returns
        -------
        list
            List of Variables from internal models.
        """
        ret = []
        for node in self.nodes:
            ret.extend(node.get_parameters_to_serialize())
        return ret

    def get_output_tensors(self):
        """Get Tensor s which represent the output of each layer of this model

        Returns
        -------
        list
            List of Tensors each of which hold output from layer
        """
        ret = []
        for node in self.nodes:
            ret.extend(node.get_output_tensors())
        return ret

    def get_update_operations(self):
        """Get update opretaions from each layer

        Returns
        -------
        list
            List of update operations from each layer
        """
        ret = []
        for node in self.nodes:
            update = node.get_update_operation()
            if update:
                ret.append(update)
        return ret
