from tensorflow.keras.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, EarlyStopping, \
    ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from pyket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, MCMCStats, \
    default_wave_function_stats_callbacks_factory
from pyket.layers import VectorToComplexNumber, ToFloat32, LogSpaceComplexNumberHistograms
from pyket.operators import Ising, cube_shape
from pyket.optimization import VariationalMonteCarlo, energy_gradient_loss
from pyket.samplers import MetropolisHastingsLocal

inputs = Input(shape=(4, 4), dtype='int8')
x = ToFloat32()(inputs)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Flatten()(x)
x = Dense(2)(x)
x = Reshape((1, 2))(x)  # keras fit_generator expect output with shape (batch_size, 1) and not simply (batch_size, )
predictions = VectorToComplexNumber()(x)
predictions = LogSpaceComplexNumberHistograms(name='psi')(predictions)
model = Model(inputs=inputs, outputs=predictions)

batch_size = 128
steps_per_epoch = 500

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=energy_gradient_loss)
model.summary()
hilbert_state_shape = cube_shape(number_of_spins_in_each_dimention=4, cube_dimention=2)
operator = Ising(h=3.0, hilbert_state_shape=hilbert_state_shape, pbc=False)
sampler = MetropolisHastingsLocal(model, batch_size, num_of_chains=16, unused_sampels=16)
monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)

checkpoint = ModelCheckpoint('ising_fcnn.h5', monitor='energy/energy', save_best_only=True, save_weights_only=True)
tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/run_0',
                                                     generator=monte_carlo_generator,
                                                     update_freq=1, histogram_freq=1,
                                                     batch_size=batch_size, write_output=False)
early_stopping = EarlyStopping(monitor='energy/relative_error', min_delta=1e-5)

callbacks = default_wave_function_stats_callbacks_factory(monte_carlo_generator,
                                                          true_ground_state_energy=-50.18662388277671) + [
                MCMCStats(monte_carlo_generator), checkpoint, tensorboard, early_stopping, TerminateOnNaN()]

model.fit_generator(monte_carlo_generator(), steps_per_epoch=steps_per_epoch, epochs=80, callbacks=callbacks,
                    max_queue_size=0, workers=0)
model.save_weights('final_ising_fcnn.h5')
