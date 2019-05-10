from keras.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam, TFOptimizer

from pyket.callbacks import TensorBoard
from pyket.callbacks.monte_carlo import TensorBoardWithGeneratorValidationData, MCMCStats, LocalEnergyStats, LocalStats, SigmaZStats, OperatorStats, WaveFunctionValuesCache
# from pyket.callbacks.exact import LocalEnergyStats, LocalStats, AbsoluteSigmaZ, OperatorStats, WaveFunctionValuesCache
from pyket.evaluation import evaluate, exact_evaluate
from pyket.layers import VectorToComplexNumber, ToFloat32, ToComplex64, PeriodicPadding, ComplexConv1D, ComplexConv2D, LogSpaceComplexNumberHistograms
from pyket.machines import RBM, DBM, SimpleConvNetAutoregressive1D, ConvNetAutoregressive2D, ResNet18, make_obc_invariants, make_pbc_invariants
from pyket.operators import NetketOperatorWrapper, Ising, Heisenberg, cube_shape
from pyket.optimization import ExactVariational, VariationalMonteCarlo, energy_gradient_loss, energy_plus_sigma_z_square_loss
from pyket.optimizers import convert_to_accumulate_gradient_optimizer, StochasticReconfiguration
from pyket.samplers import MetropoliceLocal, MetropoliceHamiltonian, AutoregressiveSampler, FastAutoregressiveSampler, Ensemble 


inputs = Input(shape=(12, 12), dtype='int8')

# Define Neural Network Machine
x = ToFloat32()(inputs)
# x = Reshape((12,12,1))(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Flatten()(x)
x = Dense(2)(x)
predictions = VectorToComplexNumber()(x)
predictions = LogSpaceComplexNumberHistograms(name='psi')(predictions)

# rbm = RBM(inputs, sym=True, alpha=1.0)
# predictions, ok = rbm.predictions, rbm.predictions_jacobian
# convnet = ConvNetAutoregressive2D(inputs, depth=10, num_of_channels=32, kernel_size=(3,3))
# predictions, conditional_log_probs, fast_sampling = convnet.predictions, convnet.conditional_log_probs, convnet.samples

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
logical_batch_size = 1024
batch_size = 256
logical_steps_per_epoch = 500
logical_actual_ratio = int(logical_batch_size / batch_size)
steps_per_epoch = logical_steps_per_epoch * logical_actual_ratio

# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=Taccumulate_sum_or_meanrue)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
convert_to_accumulate_gradient_optimizer(optimizer, update_params_frequency=logical_actual_ratio, accumulate_sum_or_mean=False)
model.compile(optimizer=optimizer, loss=energy_gradient_loss)
hilbert_state_shape = cube_shape(number_of_spins_in_each_dimention=12, cube_dimention=2)
operator = Ising(h=3.0, hilbert_state_shape=hilbert_state_shape, pbc=False)
# import netket as nk
# g = nk.graph.Hypercube(length=12, n_dim=2, pbc=False)
# hi = nk.hilbert.Spin(s=0.5, graph=g)
# netket_operator = nk.operator.Ising(h=3.0, hilbert=hi, J=1.0)
# operator = NetketOperator(netket_operator=netket_operator, hilbert_state_shape=hilbert_state_shape=hilbert_state_shape=hilbert_state_shape, max_num_of_local_connections=200)
wave_function_cache = WaveFunctionValuesCache(reset_cache_interval=logical_actual_ratio)
# VariationalMonteCarlo
sampler = MetropoliceLocal(model, batch_size, num_of_chains=10, unused_sampels=100)
# sampler = FastAutoregressiveSampler(fast_sampling, buffer_size=5000)
# sampler = FastAutoregressiveSampler(conditional_log_probs)
generator = VariationalMonteCarlo(model, operator, sampler, cache=wave_function_cache)
#### Exact Grads ####
# generator = ExactVariational(model, operator, batch_size, cache=wave_function_cache)
checkpoint = ModelCheckpoint('ising_fcnn.h5', monitor='energy', save_best_only=True, save_weights_only=True)
# tensorboard = TensorBoard(update_freq=1)
tensorboard = TensorBoardWithGeneratorValidationData(generator=generator, update_freq=1, histogram_freq=1, 
                                                     batch_size=batch_size)
early_stopping = EarlyStopping(monitor='relative_energy_error', min_delta=1e-5)
callbacks = [LocalEnergyStats(generator, true_ground_state_energy=-457.0416241), 
            wave_function_cache, LocalStats("Energy Again", sampler=sampler, operator=operator, cache=wave_function_cache), 
            SigmaZStats(generator=generator), checkpoint, tensorנoard, early_stopping, TerminateOnNaN()]
model.fit_generator(generator(), steps_per_epoch=steps_per_epoch, epochs=80, callbacks=callbacks, max_queue_size=0)
model.save_weights('final_ising_fcnn.h5')

evaluation_inputs = Input(shape=(12, 12), dtype='int8')
invariant_model = make_obc_invariants(evaluation_inputs, model)
sampler = MetropoliceLocal(invariant_model, batch_size=125, num_of_chains=10, unused_sampels=100)
generator = VariationalMonteCarlo(invariant_model, operator, sampler)
evaluate(generator(), steps=800, callbacks=callbacks[:4], keys_to_progress_bar_mapping={'energy/energy' : 'energy', 'energy/relative_error': 'relative_error'})