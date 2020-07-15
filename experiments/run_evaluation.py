import argparse

from tensorflow.keras.layers import Input
from flowket.callbacks.monte_carlo import default_wave_function_stats_callbacks_factory
from flowket.optimization import VariationalMonteCarlo
from flowket.machines.ensemble import make_2d_obc_invariants, make_up_down_invariant
from flowket.evaluation import evaluate

from train import build_model


def run(operator, config, true_ground_state_energy=None):
    model, sampler = build_model(operator, config)
    model.load_weights(config.weights_path)

    evaluation_inputs = Input(shape=config.hilbert_state_shape, dtype='int8')
    obc_input = Input(shape=config.hilbert_state_shape, dtype=evaluation_inputs.dtype)
    invariant_model = make_2d_obc_invariants(obc_input, model)
    invariant_model = make_up_down_invariant(evaluation_inputs, invariant_model)
    sampler = sampler.copy_with_new_batch_size(config.mini_batch_size)
    variational_monte_carlo = VariationalMonteCarlo(invariant_model, operator, sampler, mini_batch_size=config.mini_batch_size)
    callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo, log_in_batch_or_epoch=False, true_ground_state_energy=true_ground_state_energy)

    results = evaluate(variational_monte_carlo, steps=(config.num_of_samples) // config.mini_batch_size, callbacks=callbacks[:4],
            keys_to_progress_bar_mapping={'energy/energy': 'energy', 'energy/relative_error': 'relative_error', 'energy/local_energy_variance': 'variance'})
    print(results)


def create_evaluation_config_parser(depth, mini_batch_size, hilbert_state_shape, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_samples", default=2**16, type=int)
    parser.add_argument("--weights_path", required=True, type=str)
    parser.add_argument("--hilbert_state_shape", default=hilbert_state_shape, nargs='+', type=int)
    parser.add_argument("--width", default=32, type=int, help="num of channels at each conv layer.")
    parser.add_argument("--depth", default=depth, type=int, help="num of layers.")
    parser.add_argument("--mini_batch_size", default=mini_batch_size, type=int, help="should be the maximum number of samples the model can run without running into out of memory (depened on the gpu type & model size).")
    parser.add_argument('--use_weights_normalization', dest='weights_normalization', action='store_true')
    parser.add_argument('--no_weights_normalization', dest='weights_normalization', action='store_false')
    parser.set_defaults(weights_normalization=True)
    return parser
