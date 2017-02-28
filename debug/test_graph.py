import logging
import argparse

from luchador import nn


def _parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    logging.basicConfig(level=logging.INFO)
    model = nn.get_model_config(args.config_file)
    nn.make_model(model)


if __name__ == '__main__':
    _main()
