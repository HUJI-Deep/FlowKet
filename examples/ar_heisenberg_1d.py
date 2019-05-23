from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,  Adam

from pyket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory, MCMCStats
from pyket.machines import SimpleConvNetAutoregressive1D, ComplexValueParametersSimpleConvNetAutoregressive1D
from pyket.operators import Heisenberg
from pyket.optimization import VariationalMonteCarlo, energy_gradient_loss
from pyket.samplers import FastAutoregressiveSampler

hilbert_state_shape = (20, )
inputs = Input(shape=hilbert_state_shape, dtype='int8')
# convnet = SimpleConvNetAutoregressive1D(inputs, depth=10, num_of_channels=32)
convnet = ComplexValueParametersSimpleConvNetAutoregressive1D(inputs, depth=10, num_of_channels=16)
predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)

batch_size = 1000
steps_per_epoch = 300

# optimizer = SGD(lr=0.05)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# optimizer = ComplexValueParametersStochasticReconfiguration(model, rbm.predictions_jacobian, lr=0.05, diag_shift=0.1)
model.compile(optimizer=optimizer, loss=energy_gradient_loss)
model.summary()
operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=True)
sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size)
monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)

tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/complex_ar_dilation_run_6',
                                                     generator=monte_carlo_generator, update_freq=1,
                                                     histogram_freq=1, batch_size=batch_size, write_output=False)
callbacks = default_wave_function_stats_callbacks_factory(monte_carlo_generator,
                                                          true_ground_state_energy=-35.6175461195) + [tensorboard]
model.fit_generator(monte_carlo_generator(), steps_per_epoch=steps_per_epoch, epochs=1, callbacks=callbacks,
                    max_queue_size=0, workers=0)
model.save_weights('final_1d_heisenberg.h5')
