import argparse
from flowket.operators import Heisenberg

from train import train, create_training_config_parser
from run_evaluation import run, create_evaluation_config_parser


def main():
    parser = argparse.ArgumentParser(prog='Heisenberg NAQS')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train', help='train Heisenberg model')
    parser_eval = subparsers.add_parser('eval', help='eval Heisenberg model')
    parser_train.set_defaults(func=train)
    parser_eval.set_defaults(func=run)
    create_evaluation_config_parser(parser=parser_eval, depth=20, mini_batch_size=2**7, hilbert_state_shape=[10, 10])
    create_training_config_parser(parser=parser_train, depth=20, mini_batch_size=2**11, num_epoch=[80, 370], batch_size=[2**10, 2**11], hilbert_state_shape=[10, 10], learning_rate=[1e-3,1e-3])
    parser.add_argument('--use_pbc', dest='pbc', action='store_true')
    parser.add_argument('--no_pbc', dest='pbc', action='store_false')
    parser.set_defaults(pbc=False)
    config = parser.parse_args()

    operator = Heisenberg(hilbert_state_shape=config.hilbert_state_shape, pbc=config.pbc)
    true_ground_state_energy = None
    if len(config.hilbert_state_shape) == 2 and config.hilbert_state_shape[0] == 10 and  config.hilbert_state_shape[1] == 10:
        if config.pbc:
            true_ground_state_energy = -268.61976
        else:
            true_ground_state_energy = -251.4624
    elif (not config.pbc) and len(config.hilbert_state_shape) == 2 and config.hilbert_state_shape[0] == 16 and  config.hilbert_state_shape[1] == 16:
        true_ground_state_energy = -658.9759488
    config.func(operator, config, true_ground_state_energy)
    
if __name__ == '__main__':
    main()

