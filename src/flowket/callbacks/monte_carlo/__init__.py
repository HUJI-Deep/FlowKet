from .bad_eigen_state_stopping import BadEigenStateStopping
from .generator_iterator import GeneratorIterator
from .local_energy_stats import LocalEnergyStats
from .mcmc_stats import MCMCStats
from .observable import ObservableStats
from .runtime_stats import RuntimeStats
from .tensorboard_with_generator_validation_data import TensorBoardWithGeneratorValidationData


def default_wave_function_stats_callbacks_factory(generator, validation_generator=None,
                                                  true_ground_state_energy=None,
                                                  log_in_batch_or_epoch=True,
                                                  validation_period=1):
    from ...observables.monte_carlo import SigmaZ, AbsSigmaZ
    callbacks = []
    if validation_generator is not None:
        callbacks = [GeneratorIterator(validation_generator, period=validation_period)]
    callbacks += [LocalEnergyStats(generator, validation_generator=validation_generator,
                                   true_ground_state_energy=true_ground_state_energy,
                                   log_in_batch_or_epoch=log_in_batch_or_epoch,
                                   validation_period=validation_period),
                  ObservableStats(generator, SigmaZ(), 'sigma_z', validation_generator=validation_generator,
                                  log_in_batch_or_epoch=log_in_batch_or_epoch,
                                  validation_period=validation_period),
                  ObservableStats(generator, AbsSigmaZ(), 'abs_sigma_z', validation_generator=validation_generator,
                                  log_in_batch_or_epoch=log_in_batch_or_epoch,
                                  validation_period=validation_period),
                  RuntimeStats(generator, log_in_batch_or_epoch=log_in_batch_or_epoch)]
    return callbacks
