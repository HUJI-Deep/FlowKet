from collections import OrderedDict
import itertools
import sys

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TerminateOnNaN

from pyket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory, BadEigenStateStopping
from pyket.evaluation import evaluate
from pyket.operators.j1j2 import J1J2
from pyket.machines import ConvNetAutoregressive2D
from pyket.machines.ensemble import make_2d_obc_invariants
from pyket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from pyket.samplers import FastAutoregressiveSampler

params_grid_config = {
    'width': [32],
    'depth': [5],
    'lr': [5e-3, 1e-3, 5e-4],
    'weights_normalization': [True, False]
}
run_index = int(sys.argv[-1].strip())
ks, vs = zip(*params_grid_config.items())
params_options = list(itertools.product(*vs))
chosen_v = params_options[run_index % len(params_options)]
params = dict(zip(ks, chosen_v))
print('Chosen params: %s' % str(params))

hilbert_state_shape = (4, 4)
inputs = Input(shape=hilbert_state_shape, dtype='int8')
convnet = ConvNetAutoregressive2D(inputs, depth=params['depth'], num_of_channels=params['width'],
                                  weights_normalization=params['weights_normalization'])

predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)

batch_size = 2 ** 10
steps_per_epoch = 2 ** 8
true_ground_state_energy = -30.022227800323677

optimizer = Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
model.summary()
operator = J1J2(hilbert_state_shape=hilbert_state_shape, j2=0.5, pbc=False)
sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size)
monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)

run_name = 'j1j2_4_monte_carlo_weights_normalization_%s_depth_%s_width_%s_adam_lr_%s_run_%s' % \
           (params['weights_normalization'], params['depth'], params['width'], params['lr'], run_index)
tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/%s' % run_name,
                                                     generator=monte_carlo_generator, update_freq='epoch',
                                                     histogram_freq=1, batch_size=batch_size, write_output=False)
warly_stopping = BadEigenStateStopping(true_ground_state_energy)

callbacks = default_wave_function_stats_callbacks_factory(monte_carlo_generator, log_in_batch_or_epoch=False,
                                                          true_ground_state_energy=true_ground_state_energy) + [
                tensorboard, TerminateOnNaN(), warly_stopping]
model.fit_generator(monte_carlo_generator.to_generator(), steps_per_epoch=steps_per_epoch, epochs=60,
                    callbacks=callbacks,
                    max_queue_size=0, workers=0)
model.save_weights('before_increasing_batch__%s.h5' % run_name)
if warly_stopping.stopped_epoch is not None:
    print('stopat epoch %s because of bad eigenstate' % warly_stopping.stopped_epoch)
    sys.exit()
print('incresing batchsize to 8192')

sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size * 8)
monte_carlo_generator.set_sampler(sampler)
model.fit_generator(monte_carlo_generator.to_generator(), steps_per_epoch=steps_per_epoch, epochs=80,
                    callbacks=callbacks,
                    max_queue_size=0, workers=0)
model.save_weights('final_%s.h5' % run_name)

evaluation_inputs = Input(shape=hilbert_state_shape, dtype='int8')
invariant_model = make_2d_obc_invariants(evaluation_inputs, model)
generator = VariationalMonteCarlo(invariant_model, operator, sampler)
evaluate(generator, steps=200, callbacks=callbacks[:4],
         keys_to_progress_bar_mapping={'energy/energy': 'energy', 'energy/relative_error': 'relative_error'})
