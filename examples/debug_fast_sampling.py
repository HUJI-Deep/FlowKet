import sys
import time

import tqdm
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from pyket.optimization import loss_for_energy_minimization
from pyket.machines import ConvNetAutoregressive2D
from pyket.samplers import AutoregressiveSampler, FastAutoregressiveSampler

inputs = Input(shape=(10, 10), dtype='int8')
convnet = ConvNetAutoregressive2D(inputs, depth=40, num_of_channels=32, weights_normalization=False)
predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
sampler = FastAutoregressiveSampler(conditional_log_probs_model, batch_size=2 ** 10)
