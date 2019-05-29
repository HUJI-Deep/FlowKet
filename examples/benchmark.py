import argparse
import time
import netket as nk
from pyket.operators import Heisenberg
from pyket.optimizers import ComplexValuesStochasticReconfiguration
from pyket.optimization import VariationalMonteCarlo, energy_gradient_loss
from pyket.samplers import MetropolisHastingsHamiltonian, MetropolisHastingsLocal

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,  Adam



def run_pyket(args):
    hilbert_state_shape = (args.input_size, 1)
    inputs = Input(shape=hilbert_state_shape, dtype='int8')

    predictions = x
    predictions_jacobian = None
    model = Model(inputs=inputs, outputs=predictions)
    optimizer = ComplexValuesStochasticReconfiguration(model, predictions_jacobian, 
        lr=0.05, diag_shift=0.1, iterative_solver=args.iterative_solver)
    model.compile(optimizer=optimizer, loss=energy_gradient_loss, metrics=optimizer.metrics)
    model.summary()
    operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=True)
    sampler = MetropolisHastingsHamiltonian(model, args.batch_size, operator, 
        num_of_chains=args.pyket_num_of_chains, unused_sampels=numpy.prod(hilbert_state_shape))
    monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)



def run_netket(args):
    g = nk.graph.Hypercube(length=args.input_size, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
    ha = nk.operator.Heisenberg(hilbert=hi)
    layers = (
        nk.layer.ConvolutionalHypercube(
            length=args.input_size,
            n_dim=1,
            input_channels=1,
            output_channels=args.width,
            kernel_length=args.kernel_size),
        nk.layer.Lncosh(input_size=args.width * args.input_size),) + (nk.layer.ConvolutionalHypercube(
            length=args.input_size,
            n_dim=1,
            input_channels=args.width,
            output_channels=args.width,
            kernel_length=args.kernel_size),
        nk.layer.Lncosh(input_size=args.width * args.input_size))* (args.depth - 1)
    ma = nk.machine.FFNN(hi, layers)
    ma.init_random_parameters(seed=1234, sigma=0.1)
    sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)
    op = nk.optimizer.Sgd(learning_rate=0.01)
    gs = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=args.batch_size,
        use_iterative=args.use_iterative
        diag_shift=0.01)
    gs.run(output_prefix="ffnn_test", n_iter=5, save_params_every=5)
    start_time = time.time()
    gs.run(output_prefix="ffnn_test", n_iter=args.Num_of_iterations, save_params_every=args.Num_of_iterations)
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
    parser.add_argument('use_iterative', nargs='?', default=False, type=bool, help='use iterative solver in SR')
    return parser.parse_args()


if __name__ == '__main__':
    args =parse_args
    if args.framework == 'netket':
        time_in_seconds = run_netket(args)
    elif args.framework == 'pyket':
        time_in_seconds = run_pyket(args)
    else:
        raise Exception('unknown framework')
    print('finished')
    print('%siterations take %s seconds' % (Num_of_iterations, time_in_seconds))