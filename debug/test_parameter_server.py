from __future__ import print_function
from __future__ import absolute_import

import logging
import argparse

import requests

from luchador.util import deserialize_numpy_array
import luchador.nn as nn
from luchador.agent.rl.q_learning import DeepQLearning

_LG = logging.getLogger(__name__)


def _build_model():
    model_def = nn.get_model_config(
        'vanilla_dqn', n_actions=4, input_shape='[null, 4, 84, 84]')
    dql = DeepQLearning(
        q_learning_config={
            'discount_rate': 0.99,
            'min_reward': -1.0,
            'max_reward': 1.0,
        },
        cost_config={
            'typename': 'SSE2',
            'args': {
                'min_delta': -1.0,
                'max_delta': 1.0,
            },
        },
        optimizer_config={
            'typename': 'RMSProp',
            'args': {
                'decay': 0.95,
                'epsilon': 1e-6,
                'learning_rate': 2.5e-4,
            },
        },
    )
    dql.build(model_def, None)
    return dql


def _parse_command_line_args():
    parser = argparse.ArgumentParser(
        description='Launch environment via manager'
    )
    parser.add_argument('--port', default=5000)
    return parser.parse_args()


def _test_fetch(session, dql, port=5000):
    variables = [
        var.name for var in
        dql.models['model_0'].get_parameter_variables()
    ]

    res = session.post(
        'http://localhost:{}/fetch'.format(port),
        json={'variables': variables},
    )
    if res.status_code == 200:
        for key, data in res.json().items():
            print(key)
            print(deserialize_numpy_array(data))
    else:
        raise RuntimeError('Failed to fetch.')


def _main():
    args = _parse_command_line_args()
    dql = _build_model()
    session = requests.Session()
    _test_fetch(session, dql, args.port)


if __name__ == '__main__':
    _main()
