from .local_energy import ExactLocalEnergy
from .machine_updated import MachineUpdated
from .runtime_stats import RuntimeStats
from .sigma_z import ExactSigmaZ


def default_wave_function_callbacks_factory(generator, true_ground_state_energy=None, log_in_batch_or_epoch=True):
    return [ExactLocalEnergy(generator, true_ground_state_energy=true_ground_state_energy,
                             log_in_batch_or_epoch=log_in_batch_or_epoch),
            ExactSigmaZ(generator=generator, log_in_batch_or_epoch=log_in_batch_or_epoch),
            RuntimeStats(generator, log_in_batch_or_epoch=log_in_batch_or_epoch)]
