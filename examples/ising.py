import argparse
import sys
import itertools
import os
import pickle
import math

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

from flowket.callbacks import CheckpointByTime
from flowket.callbacks.checkpoint import load_optimizer_weights
from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from flowket.evaluation import evaluate
from flowket.layers import LogSpaceComplexNumberHistograms
from flowket.machines import ConvNetAutoregressive2D
from flowket.operators import Ising
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import FastAutoregressiveSampler
from flowket.machines.ensemble import make_2d_obc_invariants, make_up_down_invariant
from flowket.optimizers import convert_to_accumulate_gradient_optimizer


def build_model(hilbert_state_shape, depth, width, weights_normalization, learning_rate):
    inputs = Input(shape=hilbert_state_shape, dtype='int8')
    convnet = ConvNetAutoregressive2D(inputs, depth=depth, 
        num_of_channels=width, 
        weights_normalization=weights_normalization)
    predictions, conditional_log_phobs = convnet.predictions, convnet.conditional_log_probs
    model = Model(inputs=inputs, outputs=predictions)
    conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_phobs)

    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.9)
    convert_to_accumulate_gradient_optimizer(optimizer, update_params_frequency=1, 
        accumulate_sum_or_mean=True)
    model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
    model.summary()
    sampler = FastAutoregressiveSampler(conditional_log_probs_model, 16)
    return model, sampler

true_ground_state_energy_mapping = {
        2.0: -346.982, 
        2.5: -395.654, 
        3.0: -457.039,
        3.5: -524.510,
        4.0: -593.536}

def depth_to_max_mini_batch(depth):
    mapping = {1: 2**12,
            20: 2**11,
            40: 2**10
            }
    # depend on the gpu memory capacity, the above values are for 16G GPU & width = 32
    res = 1
    for k, val in mapping.items():
        if depth >= k:
            res = min(res, val)
    return res


def restore_run_state(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        initial_epoch, _ = load_optimizer_weights(model, checkpoint_path)
    else:
        initial_epoch = 0
    return initial_epoch


def run(params, batch_size_list, epochs_list):
    run_name = 'depth_%s_width_%s_weights_normalization_%s_adam_lr_%s_gamma_%s_run_%s' % (params.depth,
        params.width,
        params.no_weights_normalization,
        params.learning_rate,
        params.gamma,
        params.run_index)
    hilbert_state_shape = (12, 12)
    model, sampler = build_model(hilbert_state_shape, 
        params.depth, 
        params.width, 
        not params.no_weights_normalization, 
        params.learning_rate)
    operator = Ising(hilbert_state_shape=hilbert_state_shape, pbc=False, h=params.gamma)
    checkpoint_path = '%s.h5' % run_name
    initial_epoch = restore_run_state(model, checkpoint_path)
    total_epochs = 0
    mini_batch_size = depth_to_max_mini_batch(params.depth)
    true_ground_state_energy = true_ground_state_energy_mapping[params.gamma]

    for idx, (batch_size, epochs) in enumerate(zip(batch_size_list, epochs_list)):
        total_epochs += epochs
        if total_epochs <= initial_epoch:
            continue
        validation_sampler = sampler.copy_with_new_batch_size(min(batch_size * 8, 2**15), 
            mini_batch_size=mini_batch_size)
        assert batch_size < mini_batch_size or batch_size % mini_batch_size == 0
        sampler = sampler.copy_with_new_batch_size(batch_size, mini_batch_size)
        variational_monte_carlo = VariationalMonteCarlo(model, 
            operator, 
            sampler, 
            mini_batch_size=mini_batch_size)
        validation_generator = VariationalMonteCarlo(model, 
            operator, 
            validation_sampler, 
            wave_function_evaluation_batch_size=mini_batch_size)
        model.optimizer.set_update_params_frequency(variational_monte_carlo.update_params_frequency)
        tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/%s' % run_name,
                                                             generator=variational_monte_carlo, 
                                                             update_freq='epoch',
                                                             histogram_freq=0, 
                                                             batch_size=batch_size, 
                                                             write_output=False)
        callbacks = default_wave_function_stats_callbacks_factory(
            variational_monte_carlo, 
            validation_generator=validation_generator, 
            log_in_batch_or_epoch=False, 
            true_ground_state_energy=true_ground_state_energy, 
            validation_period=3)
        callbacks += [tensorboard, CheckpointByTime(checkpoint_path, save_weights_only=True)]
        model.fit_generator(variational_monte_carlo.to_generator(), 
            steps_per_epoch=100 * variational_monte_carlo.update_params_frequency, 
            epochs=total_epochs, callbacks=callbacks, max_queue_size=0, workers=0, 
            initial_epoch=initial_epoch)
        model.save_weights('%s_stage_%s.h5' % (run_name, idx))
        initial_epoch = total_epochs

    evaluation_inputs = Input(shape=hilbert_state_shape, dtype='int8')
    obc_input = Input(shape=hilbert_state_shape, dtype=evaluation_inputs.dtype)
    invariant_model = make_2d_obc_invariants(obc_input, model)
    invariant_model = make_up_down_invariant(evaluation_inputs, invariant_model)
    mini_batch_size=mini_batch_size // 16

    sampler = sampler.copy_with_new_batch_size(mini_batch_size)
    variational_monte_carlo = VariationalMonteCarlo(invariant_model, operator, 
        sampler, mini_batch_size=mini_batch_size)
    callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo, 
        log_in_batch_or_epoch=False, true_ground_state_energy=true_ground_state_energy)

    results = evaluate(variational_monte_carlo, 
        steps=(2**15) // mini_batch_size, 
        callbacks=callbacks[:4],
        keys_to_progress_bar_mapping={'energy/energy': 'energy', 
        'energy/relative_error': 'relative_error', 
        'energy/local_energy_variance': 'variance'})
    print(results)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", help="The Gamma in the Ising model", type=float, 
        choices=[2., 2.5, 3., 3.5, 4.], required=True)
    parser.add_argument("--depth", help="num of PixelCNN blocks", type=int, default=20, required=False)
    parser.add_argument("--width", help="num of channels in each Conv layer", type=int, default=32, required=False)
    parser.add_argument("--learning_rate", help="Adam learning rate", type=float, default=1e-3, required=False)
    parser.add_argument("--no_weights_normalization", help="don't use weights_normalization",
                    action="store_true")
    parser.add_argument("--run_index", help="index for unique run name", type=int, default=1, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()
    batch_size_list = [2**7, 2**10, 2**13]
    epochs_list = [100, 100, 50]
    run(params, batch_size_list, epochs_list)
