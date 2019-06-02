from collections import OrderedDict
import itertools
import sys

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from pyket.callbacks import TensorBoard
from pyket.callbacks.exact import default_wave_function_callbacks_factory, ExactObservableCallback
from pyket.operators.j1j2 import J1J2
from pyket.operators import NetketOperatorWrapper
from pyket.machines import ConvNetAutoregressive2D
from pyket.optimization import ExactVariational, VariationalMonteCarlo, loss_for_energy_minimization
from pyket.samplers import FastAutoregressiveSampler
from pyket.optimizers import convert_to_accumulate_gradient_optimizer

import numpy
import netket


def total_spin_netket_operator(hilbert_state_shape):
    edge_colors = []
    for i in range(numpy.prod(hilbert_state_shape)):
        edge_colors.append([i, i, 1])

    g = netket.graph.CustomGraph(edge_colors)
    hi = netket.hilbert.Spin(s=0.5, graph=g)

    sigmaz = [[1, 0], [0, -1]]
    sigmax = [[0, 1], [1, 0]]
    sigmay = [[0, -1j], [1j, 0]]

    interaction = numpy.kron(sigmaz, sigmaz) + numpy.kron(sigmax, sigmax) + numpy.kron(sigmay, sigmay)

    bond_operator = [
        (interaction).tolist(),
    ]

    bond_color = [1]

    return netket.operator.GraphOperator(hi, bondops=bond_operator, bondops_colors=bond_color)


params_grid_config = {
    'width': [32],
    'depth': [5],
    'lr': [5e-3, 1e-3],
    'weights_normalization': [False, True]
}
run_index = int(sys.argv[-1].strip())
ks, vs = zip(*params_grid_config.items())
params_options = list(itertools.product(*vs))
chosen_v = params_options[run_index % len(params_options)]
params = dict(zip(ks, chosen_v))
print('Chosen params: %s' % str(params))

hilbert_state_shape = (4, 4)
inputs = Input(shape=hilbert_state_shape, dtype='int8')
convnet = ConvNetAutoregressive2D(inputs, depth=params['depth'], num_of_channels=params['width'],
                                  weights_normalization=params['weights_normalization'])

predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)

batch_size = 2 ** 12
# For fair comparison with monte carlo eacg epoch see 2 ** 18 sampels
steps_per_epoch = 2 ** 6
true_ground_state_energy = -30.022227800323677

operator = J1J2(hilbert_state_shape=hilbert_state_shape, j2=0.5, pbc=False)
exact_variational = ExactVariational(model, operator, batch_size)

optimizer = Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999)
convert_to_accumulate_gradient_optimizer(
    optimizer,
    update_params_frequency=exact_variational.num_of_batch_until_full_cycle,
    accumulate_sum_or_mean=True)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
model.summary()

total_spin = NetketOperatorWrapper(total_spin_netket_operator(hilbert_state_shape), hilbert_state_shape)

run_name = 'j1j2_4_exact_weights_normalization_%s_depth_%s_width_%s_adam_lr_%s_run_%s' % \
           (params['weights_normalization'], params['depth'], params['width'], params['lr'], run_index)
tensorboard = TensorBoard(log_dir='tensorboard_logs/%s' % run_name,
                          update_freq='epoch',
                          write_output=False)
callbacks = default_wave_function_callbacks_factory(exact_variational, log_in_batch_or_epoch=False,
                                                    true_ground_state_energy=true_ground_state_energy) + [
                ExactObservableCallback(exact_variational, total_spin, 'total_spin', log_in_batch_or_epoch=False),
                tensorboard]
model.fit_generator(exact_variational.to_generator(), steps_per_epoch=steps_per_epoch, epochs=1000, callbacks=callbacks,
                    max_queue_size=0, workers=0)
model.save_weights('final_%s.h5' % run_name)
