import argparse
import time
import numpy
import netket as nk

from pyket.utils.jacobian import predictions_jacobian as get_predictions_jacobian
from pyket.layers import ToComplex64, ComplexConv1D, PeriodicPadding
from pyket.layers.complex.tensorflow_ops import lncosh
from pyket.operators import Heisenberg
from pyket.optimizers import ComplexValuesStochasticReconfiguration
from pyket.optimization import VariationalMonteCarlo, energy_gradient_loss
from pyket.samplers import MetropolisHastingsHamiltonian

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.ops.parallel_for import gradients


def run_pyket(args):
    hilbert_state_shape = (args.input_size, 1)
    padding = ((0, args.kernel_size - 1),)
    inputs = Input(shape=hilbert_state_shape, dtype='int8')
    x = ToComplex64()(inputs)
    for i in range(args.depth):
        x = PeriodicPadding(padding)(x)
        x = ComplexConv1D(args.width, args.kernel_size, use_bias=False)(x)
        x = Activation(lncosh)(x)
    x = Flatten()(x)
    predictions = Lambda(lambda y: tf.reduce_sum(y, axis=1, keepdims=True))(x)
    model = Model(inputs=inputs, outputs=predictions)
    # predictions_jacobian = lambda x: get_predictions_jacobian(keras_model=model)
    predictions_jacobian = lambda x: gradients.jacobian(tf.real(model.output), x, use_pfor=False)
    optimizer = ComplexValuesStochasticReconfiguration(model, predictions_jacobian,
                                                       lr=0.05, diag_shift=0.1, iterative_solver=args.use_iterative)
    model.compile(optimizer=optimizer, loss=energy_gradient_loss, metrics=optimizer.metrics)
    model.summary()
    operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=True)
    sampler = MetropolisHastingsHamiltonian(model, args.batch_size, operator,
                                            num_of_chains=args.pyket_num_of_chains,
                                            unused_sampels=numpy.prod(hilbert_state_shape))
    monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)
    model.fit_generator(monte_carlo_generator(), steps_per_epoch=5, epochs=1, max_queue_size=0, workers=0)
    start_time = time.time()
    model.fit_generator(monte_carlo_generator(), steps_per_epoch=args.num_of_iterations, epochs=1, max_queue_size=0,
                        workers=0)
    end_time = time.time()
    return end_time - start_time


def run_netket(args):
    g = nk.graph.Hypercube(length=args.input_size, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
    ha = nk.operator.Heisenberg(hilbert=hi)
    middle_layer = (nk.layer.ConvolutionalHypercube(length=args.input_size,
                                                    n_dim=1,
                                                    input_channels=args.width,
                                                    output_channels=args.width,
                                                    kernel_length=args.kernel_size),
                     nk.layer.Lncosh(input_size=args.width * args.input_size))
    middle_layers = middle_layer * (args.depth - 1)
    first_layer = (nk.layer.ConvolutionalHypercube(length=args.input_size,
                                                   n_dim=1,
                                                   input_channels=1,
                                                   output_channels=args.width,
                                                   kernel_length=args.kernel_size),
                 nk.layer.Lncosh(input_size=args.width * args.input_size),)
    ma = nk.machine.FFNN(hi, first_layer + middle_layers)
    ma.init_random_parameters(seed=1234, sigma=0.1)
    sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)
    op = nk.optimizer.Sgd(learning_rate=0.01)
    gs = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=args.batch_size,
        use_iterative=args.use_iterative,
        diag_shift=0.01)
    gs.run(output_prefix="ffnn_test", n_iter=5, save_params_every=5)
    start_time = time.time()
    gs.run(output_prefix="ffnn_test", n_iter=args.num_of_iterations, save_params_every=args.num_of_iterations)
    end_time = time.time()
    return end_time - start_time


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark settings.')
    parser.add_argument('framework', choices=['pyket', 'netket'])
    parser.add_argument('input_size', nargs='?', default=20, type=int, help='Number of spins in the input')
    parser.add_argument('batch_size', nargs='?', default=1000, type=int, help='The batch size in each iteration')
    parser.add_argument('kernel_size', nargs='?', default=4, type=int, help='The kernel size of each conv layer')
    parser.add_argument('depth', nargs='?', default=2, type=int, help='Num of conv layers before sum pooling')
    parser.add_argument('width', nargs='?', default=4, type=int, help='Num of output channels in eachconv layer')
    parser.add_argument('pyket_num_of_chains', nargs='?', default=20, type=int, help='Num of parralel mcmc in pyket')
    parser.add_argument('num_of_iterations', nargs='?', default=20, type=int, help='Num of iterations to benchmark')
    parser.add_argument('use_iterative', nargs='?', default=False, type=bool, help='use iterative solver in SR')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.framework == 'netket':
        time_in_seconds = run_netket(args)
    elif args.framework == 'pyket':
        time_in_seconds = run_pyket(args)
    else:
        raise Exception('unknown framework')
    print('finished')
    print('%s iterations take %s seconds' % (args.num_of_iterations, time_in_seconds))
