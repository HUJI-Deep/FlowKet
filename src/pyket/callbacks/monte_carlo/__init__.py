from .generator_iterator import GeneratorIterator
from .local_energy_stats import LocalEnergyStats
from .mcmc_stats import MCMCStats
from .runtime_stats import RuntimeStats
from .sigma_z_stats import SigmaZStats
from .tensorboard_with_generator_validation_data import TensorBoardWithGeneratorValidationData


def default_wave_function_stats_callbacks_factory(generator, validation_generator=None, true_ground_state_energy=None):
	callbacks = []
	if validation_generator is not None:
		callbacks = [GeneratorIterator(validation_generator())]
	callbacks += [LocalEnergyStats(generator, validation_generator=validation_generator, true_ground_state_energy=true_ground_state_energy), 
            SigmaZStats(generator=generator, validation_generator=validation_generator), RuntimeStats(generator)]
	return callbacks