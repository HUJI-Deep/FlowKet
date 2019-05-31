from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from pyket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from pyket.layers import LogSpaceComplexNumberHistograms
from pyket.machines import ConvNetAutoregressive2D
from pyket.operators import Ising
from pyket.optimization import VariationalMonteCarlo, energy_gradient_loss
from pyket.samplers import AutoregressiveSampler

hilbert_state_shape = [4, 4]
inputs = Input(shape=hilbert_state_shape, dtype='int8')
convnet = ConvNetAutoregressive2D(inputs, depth=5, num_of_channels=32, weights_normalization=False)
predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
predictions = LogSpaceComplexNumberHistograms(name='psi')(predictions)
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)

batch_size = 128
steps_per_epoch = 500

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=energy_gradient_loss)
model.summary()
operator = Ising(h=3.0, hilbert_state_shape=hilbert_state_shape, pbc=False)
sampler = AutoregressiveSampler(conditional_log_probs_model, batch_size)
variational_monte_carlo = VariationalMonteCarlo(model, operator, sampler)

validation_sampler = AutoregressiveSampler(conditional_log_probs_model, batch_size * 16)
validation_generator = VariationalMonteCarlo(model, operator, validation_sampler)

tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/2d_monte_carlo_batch_%s_run_1' % batch_size,
                                                     generator=variational_monte_carlo, update_freq=1,
                                                     histogram_freq=1, batch_size=batch_size, write_output=False)
callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo,
                                                          validation_generator=validation_generator,
                                                          true_ground_state_energy=-50.18662388277671) + [tensorboard]
model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=steps_per_epoch, epochs=8, callbacks=callbacks,
                    max_queue_size=0, workers=0)
model.save_weights('final_2d_ising_fcnn.h5')
