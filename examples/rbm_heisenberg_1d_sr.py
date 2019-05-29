from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,  Adam

import numpy

from pyket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory, MCMCStats
from pyket.machines import RBMSym
from pyket.operators import Heisenberg
from pyket.optimizers import ComplexValuesStochasticReconfiguration
from pyket.optimization import VariationalMonteCarlo, energy_gradient_loss
from pyket.samplers import MetropolisHastingsHamiltonian, MetropolisHastingsLocal

hilbert_state_shape = (20, 1)
inputs = Input(shape=hilbert_state_shape, dtype='int8')
rbm = RBMSym(inputs, stddev=0.01, use_float64_ops=True)
predictions = rbm.predictions
model = Model(inputs=inputs, outputs=predictions)

batch_size = 1000
steps_per_epoch = 300

optimizer = ComplexValuesStochasticReconfiguration(model, rbm.predictions_jacobian, lr=0.05, diag_shift=0.1,
                                                   iterative_solver=False)
model.compile(optimizer=optimizer, loss=energy_gradient_loss, metrics=optimizer.metrics)
model.summary()
operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=True)
sampler = MetropolisHastingsHamiltonian(model, batch_size, operator, num_of_chains=20, unused_sampels=numpy.prod(hilbert_state_shape))
monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)

tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/rbm_with_sr_run_6',
                                                     generator=monte_carlo_generator, update_freq=1,
                                                     histogram_freq=1, batch_size=batch_size, write_output=False)
callbacks = default_wave_function_stats_callbacks_factory(monte_carlo_generator,
                                                          true_ground_state_energy=-35.6175461195) + [MCMCStats(monte_carlo_generator), tensorboard]
model.fit_generator(monte_carlo_generator(), steps_per_epoch=steps_per_epoch, epochs=1, callbacks=callbacks,
                    max_queue_size=0, workers=0)
model.save_weights('final_1d_heisenberg.h5')
