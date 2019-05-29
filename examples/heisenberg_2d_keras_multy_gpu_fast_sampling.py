import sys

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from pyket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from pyket.layers import LogSpaceComplexNumberHistograms
from pyket.machines import ConvNetAutoregressive2D
from pyket.operators import Heisenberg
from pyket.optimization import VariationalMonteCarlo, energy_gradient_loss
from pyket.samplers import FastAutoregressiveSampler

run_index = int(sys.argv[-1].strip())
num_gpus = [1,2,4][run_index % 3]
device = tf.device("/gpu:0") if num_gpus == 1 else tf.device("/cpu:0")

batch_size = 1024
steps_per_epoch = 50


with device:
    inputs = Input(shape=(10, 10), dtype='int8')
    convnet = ConvNetAutoregressive2D(inputs, depth=5, num_of_channels=32, weights_normalization=False)
    predictions, conditional_log_phobs = convnet.predictions, convnet.conditional_log_probs
    predictions = LogSpaceComplexNumberHistograms(name='psi')(predictions)
    orig_model = Model(inputs=inputs, outputs=predictions)
    conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_phobs)
with tf.device("/gpu:0"):
    sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size)
    validation_sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size * 8)
if num_gpus > 1:
    model = multi_gpu_model(orig_model, gpus=num_gpus)
    conditional_log_probs_model = multi_gpu_model(conditional_log_probs_model, gpus=num_gpus)
else:
    model = orig_model

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=energy_gradient_loss)
model.summary()
conditional_log_probs_model.summary()
hilbert_state_shape = (10, 10)
operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=False)
monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)

validation_generator = VariationalMonteCarlo(model, operator, validation_sampler)

run_name = 'heisenberg_2d_%s_gpus_run_%s' % (num_gpus, run_index)

tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/%s' % run_name,
                                                     generator=monte_carlo_generator, update_freq=1,
                                                     histogram_freq=5, batch_size=batch_size, write_output=False)
callbacks = default_wave_function_stats_callbacks_factory(monte_carlo_generator,
                                                          validation_generator=validation_generator,
                                                          true_ground_state_energy=-251.4624) + [tensorboard]
model.fit_generator(monte_carlo_generator(), steps_per_epoch=steps_per_epoch, epochs=1, callbacks=callbacks,
                    max_queue_size=0, workers=0)
orig_model.save_weights('final_%s.h5' % run_name)
