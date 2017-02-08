from __future__ import absolute_import

import logging

import luchador.util
import luchador.nn as nn
from luchador.agent.rl.q_learning import DeepQLearning

_LG = logging.getLogger(__name__)


def _main(port=5000):
    logging.basicConfig(level=logging.INFO)
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
    app = nn.create_parameter_server_app(dql.session)
    server = luchador.util.create_server(app)
    _LG.info('Starting server on port %d', port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        _LG.info('Server on port %d stopped.', port)


if __name__ == '__main__':
    _main()
