from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from flowket.callbacks import TensorBoard
from flowket.callbacks.exact import default_wave_function_callbacks_factory
from flowket.machines import SimpleConvNetAutoregressive1D
from flowket.operators import Ising
from flowket.optimization import ExactVariational, loss_for_energy_minimization
from flowket.optimizers import convert_to_accumulate_gradient_optimizer

hilbert_state_shape = [16, ]
inputs = Input(shape=hilbert_state_shape, dtype='int8')
convnet = SimpleConvNetAutoregressive1D(inputs, depth=7, num_of_channels=32,
                                        weights_normalization=False)
model = Model(inputs=inputs, outputs=convnet.predictions)

batch_size = 2 ** 12
steps_per_epoch = 500 * (2 ** 4)

operator = Ising(h=3.0, hilbert_state_shape=hilbert_state_shape, pbc=False)
exact_variational = ExactVariational(model, operator, batch_size)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
convert_to_accumulate_gradient_optimizer(
    optimizer,
    update_params_frequency=exact_variational.num_of_batch_until_full_cycle,
    accumulate_sum_or_mean=True)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
model.summary()

# still can use monte carlo generator for estimating histograms tensorboard = TensorBoardWithGeneratorValidationData(
# log_dir='tensorboard_logs/exact_run_0', generator=monte_carlo_generator, update_freq=1, histogram_freq=1,
# batch_size=batch_size, write_output=False)
tensorboard = TensorBoard(log_dir='tensorboard_logs/exact_run_single_gpu', update_freq=1)

callbacks = default_wave_function_callbacks_factory(exact_variational,
                                                    true_ground_state_energy=-49.257706531889006) + [tensorboard]
model.fit_generator(exact_variational.to_generator(), steps_per_epoch=steps_per_epoch, epochs=2, callbacks=callbacks, max_queue_size=0,
                    workers=0)
model.save_weights('final_1d_ising_fcnn.h5')
