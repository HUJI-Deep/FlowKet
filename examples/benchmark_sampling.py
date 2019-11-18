import sys
import time

import tqdm
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from flowket.optimization import loss_for_energy_minimization
from flowket.machines import ConvNetAutoregressive2D
from flowket.samplers import AutoregressiveSampler, FastAutoregressiveSampler

run_index = int(sys.argv[-1].strip())

inputs = Input(shape=(10, 10), dtype='int8')
convnet = ConvNetAutoregressive2D(inputs, depth=10, num_of_channels=32, weights_normalization=False)
predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
sampler_cls = AutoregressiveSampler if run_index % 2 == 0 else FastAutoregressiveSampler
sampler = sampler_cls(conditional_log_probs_model, batch_size=2 ** 10)


def sample(steps):
    for _ in tqdm.trange(steps):
        next(sampler)


sample(5)  # warm up
start_time = time.time()
sample(20)
end_time = time.time()
print(end_time - start_time)
