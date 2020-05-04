from tensorflow.keras.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, EarlyStopping, \
    ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, MCMCStats, \
    default_wave_function_stats_callbacks_factory
from flowket.layers import VectorToComplexNumber, LogSpaceComplexNumberHistograms
from flowket.deepar.layers import ToFloat32
from flowket.operators import Ising, cube_shape
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import MetropolisHastingsLocal

hilbert_state_shape = cube_shape(number_of_spins_in_each_dimention=4, cube_dimention=2)
inputs = Input(shape=hilbert_state_shape, dtype='int8')
x = ToFloat32()(inputs)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Flatten()(x)
x = Dense(2)(x)
predictions = VectorToComplexNumber()(x)
predictions = LogSpaceComplexNumberHistograms(name='psi')(predictions)
model = Model(inputs=inputs, outputs=predictions)

batch_size = 128
steps_per_epoch = 500

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
model.summary()
operator = Ising(h=3.0, hilbert_state_shape=hilbert_state_shape, pbc=False)
sampler = MetropolisHastingsLocal(model, batch_size, num_of_chains=16, unused_sampels=16)
variational_monte_carlo = VariationalMonteCarlo(model, operator, sampler)

checkpoint = ModelCheckpoint('ising_fcnn.h5', monitor='energy/energy', save_best_only=True, save_weights_only=True)
tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/run_0',
                                                     generator=variational_monte_carlo,
                                                     update_freq=1, histogram_freq=1,
                                                     batch_size=batch_size, write_output=False)
early_stopping = EarlyStopping(monitor='energy/relative_error', min_delta=1e-5)

callbacks = default_wave_function_stats_callbacks_factory(variational_monte_carlo,
                                                          true_ground_state_energy=-50.18662388277671) + [
                MCMCStats(variational_monte_carlo), checkpoint, tensorboard, early_stopping, TerminateOnNaN()]

model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=steps_per_epoch, epochs=4,
                    callbacks=callbacks, max_queue_size=0, workers=0)
model.save_weights('final_ising_fcnn.h5')
