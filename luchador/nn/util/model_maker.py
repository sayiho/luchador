"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

import luchador.nn
from ..base import BaseCost, BaseLayer
from ..model import Sequential, Graph, Container
from .getter import get_input, get_layer, get_tensor, get_node

_LG = logging.getLogger(__name__)


def _get_existing_variable(name):
    """Search an existing Variable in current scope or global scope"""
    scope = luchador.nn.get_variable_scope()
    with luchador.nn.variable_scope(scope, reuse=True):
        try:
            return luchador.nn.get_variable('{}/{}'.format(scope.name, name))
        except ValueError:
            pass
        return luchador.nn.get_variable(name)


def make_io_node(config):
    """Make/fetch ``Input``/``Tensor`` instances from configuration.

    This function was itroduced to facilitate model construction from YAML.

    Parameters
    ----------
    config : [list of] dict
        typename : str
            Either ``Input`` or ``Tensor``.
        reuse : bool
            When typename is ``Input``, this value determines either the to
            reuse an existing instance of Input class or, to create new
            instance.
        args : dict
            Constructor agruments for creating new Input instance.
            See :any:`BaseInput` for detail.
        name : str
            When reusing ``Input`` or ``Tensor``, existing instance is
            retrieved using this name.

    Notes
    -----
    When fetching ``Tensor`` or an existing ``Input`` isntance with reuse, they
    must be instantiated before this function is called.

    Examples
    --------
    To create new Input

    >>> input_config1 = {
    >>>     'typename': 'Input',
    >>>     'args': {
    >>>         'dtype': 'uint8',
    >>>         'shape': [32, 5],
    >>>         'name': 'input_state',
    >>>     },
    >>> }
    >>> input1 = make_io_node(input_config1)

    then, to reuse the input,

    >>> input_config2 = {
    >>>     'typename': 'Input',
    >>>     'reuse': True,
    >>>     'name': 'input_state',
    >>> }
    >>> input2 = make_io_node(input_config2)
    >>> assert input1 is inpput2

    To reuse a Tensor from some output

    >>> input_config3 = {
    >>>     'typename': 'Tensor',
    >>>     'name': 'output_state',
    >>> }
    >>> input3 = make_io_node(input_config3)

    You can also create multiple Input instances

    >>> inputs = make_io_node([input_config2, input_config3])
    >>> assert input2 is inputs[0]
    >>> assert input3 is inputs[1]

    Returns
    -------
    [list of] ``Tensor`` or ``Input``
    """
    if isinstance(config, list):
        return [make_io_node(cfg) for cfg in config]

    type_ = config.get('typename', 'No `typename` is found.')
    if type_ not in ['Input', 'Tensor', 'Variable']:
        raise ValueError('Unexpected Input type: {}'.format(type_))

    if type_ == 'Tensor':
        ret = get_tensor(name=config['name'])
    elif type_ == 'Variable':
        ret = _get_existing_variable(name=config['name'])
    elif config.get('reuse'):
        ret = get_input(config['name'])
    else:
        ret = luchador.nn.backend.Input(**config['args'])
    return ret


def make_node(node_config):
    _LG.info('  Constructing node: %s', node_config)
    if 'typename' not in node_config:
        raise RuntimeError('Node `typename` is not given')

    node_class = get_node(node_config['typename'])
    node = node_class(**node_config.get('args', {}))

    input_ = make_io_node(node_config['input'])

    if issubclass(node_class, BaseLayer):
        node(input_)
    elif issubclass(node_class, BaseCost):
        node(*input_)

    return node


def make_layer(layer_config):
    """Make Layer instance

    Parameters
    ----------
    layer_config : dict
        typename : str
            Name of Layer class to instanciate
        args : dict
            Constructor arguments for the Layer class
        parameters: dict
            Parameter Variables to resuse. Keys are parameter keys
            and values are configuration acceptable by :func:`make_io_node`.

    Returns
    -------
    layer
        Layer object
    """
    _LG.info('  Constructing Layer: %s', layer_config)
    if 'typename' not in layer_config:
        raise RuntimeError('Layer `typename` is not given')

    layer = get_layer(layer_config['typename'])(**layer_config.get('args', {}))

    if 'parameters' in layer_config:
        parameters = OrderedDict([
            (key, make_io_node(config))
            for key, config in layer_config['parameters'].items()
        ])
        layer.set_parameter_variables(**parameters)
    return layer


def make_sequential_model(layer_configs, input_config=None):
    """Make Sequential model instance from model configuration

    Parameters
    ----------
    layer_configs : list
        ``Layer`` configuration.

    input_config : dict or None
        ``Input`` configuraiton to the model. If give, layers are built on the
        input spcified by this configuration, otherwise, model is returned
        unbuilt.

    Returns
    -------
    Model
        Resulting model
    """
    model = Sequential()
    if input_config:
        tensor = make_io_node(input_config)
    for config in layer_configs:
        layer = make_layer(config)
        if input_config:
            tensor = layer(tensor)
        model.add_layer(layer)
    return model


def make_graph_model(node_configs):
    model = Graph()
    for node_config in node_configs:
        node = make_node(node_config)
        model.add_node(node)
    return model


def make_container_model(input_config, model_configs, output_config=None):
    """Make ``Container`` model from model configuration

    Parameters
    ----------
    input_config : [list of] dict
        See :any::make_io_node

    model_config : list
        Model configuration.

    output_config : [list of] dict
        See :any::make_io_node

    Returns
    -------
    Model
        Resulting model
    """
    model = Container()
    model.input = make_io_node(input_config)
    for conf in model_configs:
        _LG.info('Building Model: %s', conf.get('name', 'No name defined'))
        model.add_model(conf['name'], make_model(conf))

    if output_config:
        model.output = make_io_node(output_config)
    return model


def make_model(model_config):
    """Make model from model configuration

    Parameters
    ----------
    model_config : [list of] dict
        Model configuration in dict or list of configurations.

    Returns
    -------
    [list of] Model
        Resulting model[s]
    """
    if isinstance(model_config, list):
        return [make_model(cfg) for cfg in model_config]

    _type = model_config.get('typename', 'No model type found')
    if _type == 'Sequential':
        return make_sequential_model(**model_config.get('args', {}))
    if _type == 'Graph':
        return make_graph_model(**model_config.get('args', {}))
    if _type == 'Container':
        return make_container_model(**model_config.get('args', {}))

    raise ValueError('Unexpected model type: {}'.format(_type))
