from __future__ import absolute_import

from .base_model import BaseModel


class Graph(BaseModel):
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)
