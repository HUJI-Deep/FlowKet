import argparse
import importlib
import json
import math
import os

import tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda
from flowket.deepar.layers import ToFloat32, WeightNormalization, ExpandInputDim, PeriodicPadding

from flowket.callbacks import CheckpointByTime
from flowket.callbacks.checkpoint import load_optimizer_weights
from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from flowket.evaluation import evaluate
from flowket.machines import ConvNetAutoregressive2D
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import FastAutoregressiveSampler
from flowket.machines.ensemble import make_2d_obc_invariants, make_up_down_invariant
from flowket.optimizers import convert_to_accumulate_gradient_optimizer

horovod_spec = importlib.util.find_spec("horovod")
horovod_found = horovod_spec is not None
if horovod_found:
    import horovod.tensorflow.keras as hvd
    from flowket.optimization.horovod_variational_monte_carlo import HorovodVariationalMonteCarlo

def init_horovod():
    hvd.init()
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tensorflow.Session(config=config))


def build_model(operator, config):
    inputs = Input(shape=operator.hilbert_state_shape, dtype='int8')
    convnet = ConvNetAutoregressive2D(inputs, depth=config.depth, num_of_channels=config.width, weights_normalization=config.weights_normalization)
    predictions, conditional_log_phobs = convnet.predictions, convnet.conditional_log_probs
    model = Model(inputs=inputs, outputs=predictions)
    conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_phobs)
    sampler = FastAutoregressiveSampler(conditional_log_probs_model, 16)
    return model, sampler


def compile_model(model, intial_learning_rate, use_horovod):
    optimizer = Adam(lr=intial_learning_rate, beta_1=0.9, beta_2=0.9)
    optimizer = convert_to_accumulate_gradient_optimizer(optimizer, update_params_frequency=1, accumulate_sum_or_mean=True, use_horovod=use_horovod)
    model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
    return optimizer


def load_weights_if_exist(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print('checkpoint %s' % checkpoint_path)
        model.load_weights(checkpoint_path)
        initial_epoch, _ = load_optimizer_weights(model, checkpoint_path)
    else:
        initial_epoch = 0
    return initial_epoch


def to_valid_stages_config(config):
    if len(config.batch_size) == 1:
        config.batch_size = [config.batch_size] * len(config.num_epoch)
    if len(config.learning_rate) == 1:
        config.learning_rate = [config.learning_rate] * len(config.num_epoch)
    assert len(config.batch_size) == len(config.num_epoch) and len(config.learning_rate) == len(config.num_epoch)


def train(operator, config, true_ground_state_energy=None):
    if config.use_horovod:
        init_horovod()
    to_valid_stages_config(config)
    is_rank_0 = (not config.use_horovod) or hvd.rank() == 0
    if is_rank_0:
        save_config(config)
    model, sampler = build_model(operator, config)
    optimizer = compile_model(model, config.learning_rate[0], config.use_horovod)
    checkpoint_path = os.path.join(config.output_path, 'model.h5')
    initial_epoch = load_weights_if_exist(model, checkpoint_path)

    total_epochs = 0

    for idx, (batch_size, num_epoch, learning_rate) in enumerate(zip(config.batch_size, config.num_epoch, config.learning_rate)):
        total_epochs += num_epoch
        if total_epochs <= initial_epoch:
            continue
        vmc_cls = VariationalMonteCarlo
        if config.use_horovod:
            batch_size = int(math.ceil(batch_size / hvd.size()))
            vmc_cls = HorovodVariationalMonteCarlo

        validation_sampler = sampler.copy_with_new_batch_size(min(batch_size * 8, 2**15), mini_batch_size=config.mini_batch_size)
        assert batch_size < config.mini_batch_size or batch_size % config.mini_batch_size == 0
        sampler = sampler.copy_with_new_batch_size(batch_size, config.mini_batch_size)
        variational_monte_carlo = vmc_cls(model,
            operator,
            sampler,
            mini_batch_size=config.mini_batch_size)
        validation_generator = vmc_cls(model,
            operator,
            validation_sampler,
            wave_function_evaluation_batch_size=config.mini_batch_size)
        optimizer.set_update_params_frequency(variational_monte_carlo.update_params_frequency)
        K.set_value(optimizer.lr, learning_rate)

        callbacks = default_wave_function_stats_callbacks_factory(
            variational_monte_carlo, validation_generator=validation_generator, log_in_batch_or_epoch=False, true_ground_state_energy=true_ground_state_energy, validation_period=config.validation_period)
        
        if config.use_horovod:
            callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), ] + callbacks + [hvd.callbacks.MetricAverageCallback()]
        if is_rank_0:
            tensorboard = TensorBoardWithGeneratorValidationData(log_dir=config.output_path,
                                                                 generator=variational_monte_carlo, update_freq='epoch',
                                                                 histogram_freq=0, batch_size=batch_size, write_output=False, write_graph=False)
            callbacks += [tensorboard, CheckpointByTime(checkpoint_path, save_weights_only=True)]
            verbose = 1
        else:
            verbose = 0
        model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=config.steps_per_epoch * variational_monte_carlo.update_params_frequency, epochs=total_epochs, callbacks=callbacks,
                        max_queue_size=0, workers=0, initial_epoch=initial_epoch, verbose=verbose)
        if is_rank_0:
            model.save_weights(os.path.join(config.output_path, 'stage_%s.h5' % (idx + 1)))
        initial_epoch = total_epochs

    evaluation_inputs = Input(shape=config.hilbert_state_shape, dtype='int8')
    obc_input = Input(shape=config.hilbert_state_shape, dtype=evaluation_inputs.dtype)
    invariant_model = make_2d_obc_invariants(obc_input, model)
    invariant_model = make_up_down_invariant(evaluation_inputs, invariant_model)
    mini_batch_size = config.mini_batch_size // 16

    sampler = sampler.copy_with_new_batch_size(config.mini_batch_size)
    
    vmc_cls = VariationalMonteCarlo
    if config.use_horovod:
        vmc_cls = HorovodVariationalMonteCarlo

    variational_monte_carlo = vmc_cls(invariant_model, operator, sampler, mini_batch_size=config.mini_batch_size)
    callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo, log_in_batch_or_epoch=False, true_ground_state_energy=true_ground_state_energy)
    if config.use_horovod:
        callbacks = callbacks + [hvd.callbacks.MetricAverageCallback()]
    results = evaluate(variational_monte_carlo, steps=(2**15) // mini_batch_size, callbacks=callbacks[:4],
            keys_to_progress_bar_mapping={'energy/energy': 'energy', 'energy/relative_error': 'relative_error', 'energy/local_energy_variance': 'variance'}, verbose=is_rank_0)
    if is_rank_0:
        print(results)


def save_config(config):
    if hasattr(config, 'func'):
        del config.func
    config_dict = vars(config)
    os.makedirs(config.output_path, exist_ok=True)
    with open(os.path.join(config.output_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)


def create_training_config_parser(depth, mini_batch_size, num_epoch, batch_size, hilbert_state_shape, learning_rate, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--validation_period", default=3, type=int, help='frequency (in epoch) of estimating energy with large batch.')
    parser.add_argument("--hilbert_state_shape", default=hilbert_state_shape, nargs='+', type=int)
    parser.add_argument("--batch_size", default=batch_size, nargs='+', 
        type=int, help='The batch size at each stage of the training.')
    parser.add_argument("--num_epoch", default=num_epoch, nargs='+', 
        type=int, help='Num of epochs at each training stage.')
    parser.add_argument("--steps_per_epoch", default=100, type=int, help="number of batch in each epoch.")
    parser.add_argument("--learning_rate", default=learning_rate, nargs='+', 
        type=float, help='The learning rate at each stage of the training.')
    parser.add_argument("--width", default=32, type=int, help="num of channels at each conv layer.")
    parser.add_argument("--depth", default=depth, type=int, help="num of layers.")
    parser.add_argument("--mini_batch_size", default=mini_batch_size, type=int, help="should be the maximum number of samples the model can run without running into out of memory (depened on the gpu type & model size).")
    parser.add_argument('--use_weights_normalization', dest='weights_normalization', action='store_true')
    parser.add_argument('--no_weights_normalization', dest='weights_normalization', action='store_false')
    parser.set_defaults(weights_normalization=True)
    parser.add_argument('--use_horovod', dest='use_horovod', action='store_true')
    parser.add_argument('--disable_horovod', dest='use_horovod', action='store_false')
    parser.set_defaults(use_horovod=False)
    return parser

