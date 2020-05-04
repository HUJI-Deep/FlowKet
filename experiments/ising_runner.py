import argparse
from flowket.operators import Ising

from train import train, create_training_config_parser
from run_evaluation import run, create_evaluation_config_parser


def main():
    parser = argparse.ArgumentParser(prog='Ising NAQS')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train', help='train ising model')
    parser_eval = subparsers.add_parser('eval', help='eval ising model')
    parser_train.set_defaults(func=train)
    parser_eval.set_defaults(func=run)

    create_evaluation_config_parser(parser=parser_eval, depth=10, mini_batch_size=2**8, hilbert_state_shape=[12, 12])
    create_training_config_parser(parser=parser_train, depth=10, mini_batch_size=2**12, num_epoch=[100, 100, 50], batch_size=[2**7, 2**10, 2**13], hilbert_state_shape=[12, 12], learning_rate=[1e-3, 1e-3, 1e-3])
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument('--use_pbc', dest='pbc', action='store_true')
    parser.add_argument('--no_pbc', dest='pbc', action='store_false')
    parser.set_defaults(pbc=False)
    config = parser.parse_args()

    operator = Ising(hilbert_state_shape=config.hilbert_state_shape, pbc=config.pbc, h=config.gamma)
    true_ground_state_energy = None
    if (not config.pbc) and len(config.hilbert_state_shape) == 2 and config.hilbert_state_shape[0] == 12 and  config.hilbert_state_shape[1] == 12:
        true_ground_state_energy =  {2.0: -346.982,
            2.5: -395.654,
            3.0: -457.039,
            3.5: -524.510,
            4.0: -593.536}[config.gamma]
    config.func(operator, config, true_ground_state_energy)

if __name__ == '__main__':
    main()

