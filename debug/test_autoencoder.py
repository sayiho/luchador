import os
import gzip
import pickle
import logging
import argparse

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'


import luchador.nn as nn
import luchador.nn.summary


def _parase_command_line_args():
    default_mnist_path = os.path.join('data', 'mnist.pkl.gz')

    parser = argparse.ArgumentParser(
        description='Test autoencoder'
    )
    parser.add_argument('model_file')
    parser.add_argument(
        '--mnist', default=default_mnist_path,
        help='Path to MNIST dataset. Default: {}'.format(default_mnist_path),
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _load_data(filepath):
    with gzip.open(filepath, 'rb') as file_:
        return pickle.load(file_)


def _initialize_logger(debug):
    from luchador.util import initialize_logger
    message_format = (
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
        if debug else '%(asctime)s: %(levelname)5s: %(message)s'
    )
    level = logging.DEBUG if debug else logging.INFO
    initialize_logger(
        name='luchador', message_format=message_format, level=level)


def _build_model(model_file, input_shape):
    model_def = nn.get_model_config(model_file, input_shape=input_shape)
    return nn.make_model(model_def)


def _main():
    args = _parase_command_line_args()
    _initialize_logger(args.debug)

    data_format = luchador.get_nn_conv_format()
    input_shape = [32, 28, 28, 1] if data_format == 'NHWC' else [32, 1, 28, 28]
    autoencoder = _build_model(args.model_file, input_shape)
    cost = autoencoder.output

    optimizer = nn.get_optimizer('Adam')(learning_rate=0.01)
    wrt = autoencoder.get_parameters_to_train()
    minimize_op = optimizer.minimize(loss=cost, wrt=wrt)
    update_op = autoencoder.get_update_operations()
    updates = update_op + [minimize_op]
    print updates

    session = nn.Session()
    session.initialize()

    summary = nn.summary.SummaryWriter(output_dir='tmp')
    if session.graph:
        summary.add_graph(session.graph)

    train_set, valid_set, test_set = _load_data(args.mnist)

    shape = [-1, 28, 28, 1] if data_format == 'NHWC' else [-1, 1, 28, 28]
    images = train_set[0].reshape(shape)

    n_batch = 32
    try:
        for i in range(0, 40000, n_batch):
            cost_ = session.run(
                inputs={autoencoder.input: images[i:i+n_batch, ...]},
                outputs=cost, updates=updates, name='opt',
            )
            print i, cost_
    except KeyboardInterrupt:
        pass

    recon = session.run(
        outputs=autoencoder.models['autoencoder'].output,
        inputs={autoencoder.input: images[:32, ...]}
    )

    import matplotlib.pyplot as plt

    img = images[0, :, :, 0] if data_format == 'NHWC' else images[0, 0, :, :]
    rec = recon[0, :, :, 0] if data_format == 'NHWC' else recon[0, 0, :, :]

    fig = plt.figure()
    axis = fig.add_subplot(2, 1, 1)
    axis.imshow(img, cmap='gray')
    axis = fig.add_subplot(2, 1, 2)
    axis.imshow(rec, cmap='gray')
    plt.show()


if __name__ == '__main__':
    _main()
