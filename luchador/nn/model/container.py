"""Define Container class which can build model stiching Sequeitla model"""
from __future__ import absolute_import

from collections import OrderedDict

from .base_model import BaseModel


class Container(BaseModel):
    """Data structure for handling multiple network architectures at once

    Using this class and build utility functions make it easy to build
    multi-branching-merging network.
    """
    def __init__(self):
        super(Container, self).__init__()
        self.models = OrderedDict()
        self.input = None
        self.output = None

    def _get_components(self):
        return self.models.values()

    def add_model(self, name, model):
        """Add model.

        Parameters
        ----------
        name : str
            Name of model to store.

        model : Model
            Model object.
        """
        self.models[name] = model
        return self

    def __repr__(self):
        return repr({self.__class__.__name__: self.models})
