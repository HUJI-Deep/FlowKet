from collections import OrderedDict
import itertools
import sys

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from flowket.machines import SimpleConvNetAutoregressive1D, ComplexValuesSimpleConvNetAutoregressive1D
from flowket.operators import Heisenberg
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import FastAutoregressiveSampler

params_grid_config = {
    'width': [16, 32],
    'depth': [5, 8],
    'lr': [1e-3, 1e-2, 5e-3],
    'complex_ops': [True, False]
}
run_index = int(sys.argv[-1].strip())
ks, vs = zip(*params_grid_config.items())
params_options = list(itertools.product(*vs))
chosen_v = params_options[run_index % len(params_options)]
params = dict(zip(ks, chosen_v))
print('Chosen params: %s' % str(params))

hilbert_state_shape = (20,)
inputs = Input(shape=hilbert_state_shape, dtype='int8')
if params['complex_ops']:
    convnet = ComplexValuesSimpleConvNetAutoregressive1D(inputs, depth=params['depth'],
                                                         num_of_channels=params['width'], max_dilation_rate=4)
else:
    convnet = SimpleConvNetAutoregressive1D(inputs, depth=params['depth'], num_of_channels=params['width'] * 2,
                                            max_dilation_rate=4, weights_normalization=False)
predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)

batch_size = 1000
steps_per_epoch = 300

optimizer = Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
model.summary()
operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=True)
sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size)
variational_monte_carlo = VariationalMonteCarlo(model, operator, sampler)

run_name = 'naqs_complex_ops_%s_dilation_depth_%s_width_%s_adam_lr_%s_run_%s' % \
           (params['complex_ops'], params['depth'], params['width'], params['lr'], run_index)
tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/%s' % run_name,
                                                     generator=variational_monte_carlo, update_freq=1,
                                                     histogram_freq=1, batch_size=batch_size, write_output=False)
callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo,
                                                          true_ground_state_energy=-35.6175461195) + [tensorboard]
model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=steps_per_epoch, epochs=15,
                    callbacks=callbacks, max_queue_size=0, workers=0)
