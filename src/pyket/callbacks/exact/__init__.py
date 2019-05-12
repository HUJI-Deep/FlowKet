from .local_energy import ExactLocalEnergy
from .runtime_stats import RuntimeStats
from .sigma_z import ExactSigmaZ


def default_wave_function_callbacks_factory(generator, true_ground_state_energy=None):
    return [ExactLocalEnergy(generator, true_ground_state_energy=true_ground_state_energy),
            ExactSigmaZ(generator=generator), RuntimeStats(generator)]
