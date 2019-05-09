from .local_energy import ExactLocalEnergy
from .sigma_z import ExactSigmaZ


def default_wave_function_callbacks_factory(generator, true_ground_state_energy=None):
	callbacks = []
	callbacks += [ExactLocalEnergy(generator, true_ground_state_energy=true_ground_state_energy), 
	ExactSigmaZ(generator=generator)]
	return callbacks