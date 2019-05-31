from .generator_iterator import GeneratorIterator
from .local_energy_stats import LocalEnergyStats
from .mcmc_stats import MCMCStats
from .runtime_stats import RuntimeStats
from .sigma_z_stats import SigmaZStats
from .tensorboard_with_generator_validation_data import TensorBoardWithGeneratorValidationData


def default_wave_function_stats_callbacks_factory(generator, validation_generator=None,
                                                  true_ground_state_energy=None,
                                                  log_in_batch_or_epoch=True):
    callbacks = []
    if validation_generator is not None:
        callbacks = [GeneratorIterator(validation_generator)]
    callbacks += [LocalEnergyStats(generator, validation_generator=validation_generator,
                                   true_ground_state_energy=true_ground_state_energy,
                                   log_in_batch_or_epoch=log_in_batch_or_epoch),
                  SigmaZStats(generator=generator, validation_generator=validation_generator,
                              log_in_batch_or_epoch=log_in_batch_or_epoch),
                  RuntimeStats(generator, log_in_batch_or_epoch=log_in_batch_or_epoch)]
    return callbacks
