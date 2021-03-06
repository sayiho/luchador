"""Module for running server which manage remote environments

Manager Server is for starting new Environment server on local machine via
POST request.
Manager server can be created using :py:func:`create_manager_app` as following.

>>> import luchador.env
>>> app = luchador.env.remote.create_manager_app(env)
>>> server = luchador.env.remote.create_server(app)
>>> server.start()

Manager API
-----------

+---------+-------------------------------------------------------------------+
|**Start new environment server**                                             |
+---------+-------------------------------------------------------------------+
|URL      |``/create``                                                        |
+---------+-------------------------------------------------------------------+
|Method   |``POST``                                                           |
+---------+-------------------------------------------------------------------+
|Data     |``environment``: Environment configuration in JSON format          |
|Params   |``port``: ``integer``                                              |
+---------+-------------------------------------------------------------------+
|Success  |**Code**: 200                                                      |
|Response |                                                                   |
|         |**Content (JSON)**                                                 |
|         | ``environment``: ``"Environment representation"``                 |
+---------+-------------------------------------------------------------------+
"""
from __future__ import absolute_import

import os
import logging
import tempfile
import subprocess

import flask

import luchador.util

_LG = logging.getLogger(__name__)


def _create_temp_environment_file(config):
    _LG.info('Creating environment config file')
    dumped = luchador.util.pprint_dict(config)
    _LG.info('\n%s', dumped)
    with tempfile.NamedTemporaryFile(delete=False) as file_:
        file_.write(dumped)
    return file_


def _parse_params(params):
    if 'environment' not in params:
        raise ValueError('Missing parameter; "environment"')

    try:
        port = str(params.get('port'))
    except ValueError:
        raise ValueError('"port" must be string type')

    xvfb = params.get('xvfb')

    return params['environment'], port, xvfb


def create_manager_app():
    """Create manager server application"""
    app = flask.Flask(__name__)
    attr = {}

    @app.route('/create', methods=['POST'])
    def _create_env():  # pylint: disable=unused-variable
        params = flask.request.get_json()

        try:
            env_cfg, port, xvfb = _parse_params(params)
        except ValueError as error:
            return error.args[0], 400

        file_ = _create_temp_environment_file(env_cfg)
        cmd = ['luchador', 'serve', 'env', file_.name, '--port', port]
        if xvfb:
            cmd = ['xvfb-run'] + cmd
        _LG.info('Starting environment server: %s', cmd)
        # For portable way to start independent process
        # see http://stackoverflow.com/a/13256908
        subprocess.Popen(cmd, preexec_fn=os.setsid)
        return 'ok'

    app.attr = attr
    return app
