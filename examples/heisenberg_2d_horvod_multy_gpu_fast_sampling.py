import sys

import horovod.tensorflow.keras as hvd
import math
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

# Horovod: initialize Horovod.
hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from flowket.layers import LogSpaceComplexNumberHistograms
from flowket.machines import ConvNetAutoregressive2D
from flowket.operators import Heisenberg
from flowket.optimization import loss_for_energy_minimization
from flowket.optimization.horovod_variational_monte_carlo import HorovodVariationalMonteCarlo
from flowket.samplers import FastAutoregressiveSampler

run_index = int(sys.argv[-1].strip())

batch_size = 1024
batch_size = int(math.ceil(batch_size / hvd.size()))
steps_per_epoch = 50

inputs = Input(shape=(10, 10), dtype='int8')
convnet = ConvNetAutoregressive2D(inputs, depth=5, num_of_channels=32, weights_normalization=False)
predictions, conditional_log_phobs = convnet.predictions, convnet.conditional_log_probs
predictions = LogSpaceComplexNumberHistograms(name='psi')(predictions)
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_phobs)
sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size)
validation_sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size * 8)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
optimizer = hvd.DistributedOptimizer(optimizer)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
model.summary()
conditional_log_probs_model.summary()
hilbert_state_shape = (10, 10)
operator = Heisenberg(hilbert_state_shape=hilbert_state_shape, pbc=False)
variational_monte_carlo = HorovodVariationalMonteCarlo(model, operator, sampler)

validation_generator = HorovodVariationalMonteCarlo(model, operator, validation_sampler)

run_name = 'horovod_fast_sampling_heisenberg_2d_%s_gpus' % (run_index)

tensorboard = TensorBoardWithGeneratorValidationData(log_dir='tensorboard_logs/%s' % run_name,
                                                     generator=variational_monte_carlo, update_freq=1,
                                                     histogram_freq=5, batch_size=batch_size, write_output=False)
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), ] + default_wave_function_stats_callbacks_factory(
    variational_monte_carlo,
    validation_generator=validation_generator, true_ground_state_energy=-251.4624) + [
                hvd.callbacks.MetricAverageCallback()]
if hvd.rank() == 0:
    callbacks += [tensorboard]
model.fit_generator(variational_monte_carlo.to_generator(), steps_per_epoch=steps_per_epoch, epochs=1, callbacks=callbacks,
                    max_queue_size=0, workers=0)
if hvd.rank() == 0:
    model.save_weights('final_%s.h5' % run_name)
