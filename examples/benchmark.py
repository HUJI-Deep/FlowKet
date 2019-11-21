import argparse
import time
import numpy
import netket as nk

from flowket.utils.jacobian import predictions_jacobian as get_predictions_jacobian
from flowket.layers import ToComplex64, ToComplex128, ComplexConv1D
from flowket.deepar.layers import PeriodicPadding
from flowket.layers.complex.tensorflow_ops import lncosh
from flowket.operators import Heisenberg
from flowket.optimizers import ComplexValuesStochasticReconfiguration
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import MetropolisHastingsHamiltonian

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.ops.parallel_for import gradients
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

K.set_floatx('float64')


def run_pyket(args):
    hilbert_state_shape = (args.input_size, 1)
    padding = ((0, args.kernel_size - 1),)
    inputs = Input(shape=hilbert_state_shape, dtype='int8')
    x = ToComplex128()(inputs)
    for i in range(args.depth):
        x = PeriodicPadding(padding)(x)
        x = ComplexConv1D(args.width, args.kernel_size, use_bias=False, dtype=tf.complex128)(x)
        x = Activation(lncosh)(x)
    x = Flatten()(x)
    predictions = Lambda(lambda y: tf.reduce_sum(y, axis=1, keepdims=True))(x)
    model = Model(inputs=inputs, outputs=predictions)
    if args.fast_jacobian:
        predictions_jacobian = lambda x: get_predictions_jacobian(keras_model=model)
    else:
        predictions_jacobian = lambda x: gradients.jacobian(tf.real(model.output), x, use_pfor=not args.no_pfor)
    if args.use_stochastic_reconfiguration:
        optimizer = ComplexValuesStochasticReconfiguration(model, predictions_jacobian,
                                                           lr=args.learning_rate, diag_shift=10.0, 
                                                           iterative_solver=args.use_iterative,
                                                           use_cholesky=args.use_cholesky,
                                                           iterative_solver_max_iterations=None)
        model.compile(optimizer=optimizer, loss=loss_for_energy_minimization, metrics=optimizer.metrics)
    else:
        optimizer = SGD(lr=args.learning_rate)
        model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
    model.summary()
    operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=True)
    sampler = MetropolisHastingsHamiltonian(model, args.batch_size, operator,
                                            num_of_chains=args.pyket_num_of_chains,
                                            unused_sampels=numpy.prod(hilbert_state_shape))
    variational_monte_carlo = VariationalMonteCarlo(model, operator, sampler)
    model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=5, epochs=1, max_queue_size=0, workers=0)
    start_time = time.time()
    model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=args.num_of_iterations, epochs=1,
                        max_queue_size=0, workers=0)
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
    op = nk.optimizer.Sgd(learning_rate=args.learning_rate)
    method = 'Sr' if args.use_stochastic_reconfiguration else 'Gd'
    gs = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        method=method,
        optimizer=op,
        n_samples=args.batch_size,
        use_iterative=args.use_iterative,
        use_cholesky=args.use_cholesky,
        diag_shift=10.0)
    gs.run(output_prefix="ffnn_test", n_iter=5, save_params_every=5)
    start_time = time.time()
    gs.run(output_prefix="ffnn_test", n_iter=args.num_of_iterations, save_params_every=args.num_of_iterations)
    end_time = time.time()
    return end_time - start_time


def define_args_parser():
    parser = argparse.ArgumentParser(description='Benchmark settings.')
    parser.add_argument('framework', choices=['flowket', 'netket'])
    parser.add_argument('-learning_rate', '-l', nargs='?', default=0.0001, type=float, help='The learning rate')
    parser.add_argument('-input_size', '-i', nargs='?', default=20, type=int, help='Number of spins in the input')
    parser.add_argument('-batch_size', '-b', nargs='?', default=1000, type=int, help='The batch size in each iteration')
    parser.add_argument('-kernel_size', '-k', nargs='?', default=4, type=int, help='The kernel size of each conv layer')
    parser.add_argument('-depth', '-d', nargs='?', default=2, type=int, help='Num of conv layers before sum pooling')
    parser.add_argument('-width', '-w', nargs='?', default=4, type=int, help='Num of output channels in eachconv layer')
    parser.add_argument('-pyket_num_of_chains', nargs='?', default=20, type=int, help='Num of parralel mcmc in flowket')
    parser.add_argument('-num_of_iterations', nargs='?', default=20, type=int, help='Num of iterations to benchmark')
    parser.add_argument('-use_cholesky', action='store_true', help='use cholesky solver in SR')
    parser.add_argument('-use_iterative',action='store_true', help='use iterative solver in SR')
    parser.add_argument('-pyket_on_cpu', '-cpu',  action='store_true', help='force running flowket on cpu')
    parser.add_argument('-use_stochastic_reconfiguration', '-sr',  action='store_true', help='Use stochastic Reconfiguration')
    parser.add_argument('-fast_jacobian', action='store_true', help='use flowket custom code for jacobian (still have bugs)')
    parser.add_argument('-no_pfor', action='store_true', help="don't use tensorflow pfor")
    return parser


def run(args):
    if args.framework == 'netket':
        time_in_seconds = run_netket(args)
    elif args.framework == 'flowket':
        if args.pyket_on_cpu:
            with tf.device('/cpu:0'):
                time_in_seconds = run_pyket(args)
        else:
            time_in_seconds = run_pyket(args)
    else:
        raise Exception('unknown framework')
    return time_in_seconds


if __name__ == '__main__':
    args = define_args_parser().parse_args()
    time_in_seconds = run(args)
    print('finished')
    print('%s iterations take %s seconds' % (args.num_of_iterations, time_in_seconds))
