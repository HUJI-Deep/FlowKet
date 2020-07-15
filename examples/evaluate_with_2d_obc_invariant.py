from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from flowket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, \
    default_wave_function_stats_callbacks_factory
from flowket.evaluation import evaluate
from flowket.layers import LogSpaceComplexNumberHistograms
from flowket.machines import ConvNetAutoregressive2D
from flowket.machines.ensemble import make_2d_obc_invariants
from flowket.operators import Ising
from flowket.optimization import VariationalMonteCarlo, loss_for_energy_minimization
from flowket.samplers import AutoregressiveSampler

hilbert_state_shape = [4, 4]
inputs = Input(shape=hilbert_state_shape, dtype='int8')
convnet = ConvNetAutoregressive2D(inputs, depth=5, num_of_channels=32, weights_normalization=False)
predictions, conditional_log_probs = convnet.predictions, convnet.conditional_log_probs
model = Model(inputs=inputs, outputs=predictions)
conditional_log_probs_model = Model(inputs=inputs, outputs=conditional_log_probs)

batch_size = 128
steps_per_epoch = 500

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss=loss_for_energy_minimization)
model.summary()
operator = Ising(h=3.0, hilbert_state_shape=hilbert_state_shape, pbc=False)
sampler = AutoregressiveSampler(conditional_log_probs_model, batch_size)
monte_carlo_generator = VariationalMonteCarlo(model, operator, sampler)

callbacks = default_wave_function_stats_callbacks_factory(monte_carlo_generator, true_ground_state_energy=-50.18662388277671)
model.fit_generator(monte_carlo_generator, steps_per_epoch=steps_per_epoch, epochs=2, callbacks=callbacks,
                    max_queue_size=0, workers=0)

print('evaluate normal model')
evaluate(monte_carlo_generator, steps=200, callbacks=callbacks,
         keys_to_progress_bar_mapping={'energy/energy': 'energy', 'energy/relative_error': 'relative_error'})

print('evaluate invariant model')
evaluation_inputs = Input(shape=hilbert_state_shape, dtype='int8')
invariant_model = make_2d_obc_invariants(evaluation_inputs, model)
monte_carlo_generator = VariationalMonteCarlo(invariant_model, operator, sampler)
callbacks = default_wave_function_stats_callbacks_factory(monte_carlo_generator, true_ground_state_energy=-50.18662388277671)
evaluate(monte_carlo_generator, steps=200, callbacks=callbacks,
         keys_to_progress_bar_mapping={'energy/energy': 'energy', 'energy/relative_error': 'relative_error'})

