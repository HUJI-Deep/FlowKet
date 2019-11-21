import functools
import numpy

from .observable import LambdaObservable


def abs_sigma_z(wave_function, configurations):
    total_spins_per_sample = numpy.prod(configurations.shape[1:])
    axis_to_sum = tuple((range(1, len(configurations.shape))))
    return numpy.absolute(configurations.sum(axis=axis_to_sum)) / total_spins_per_sample


def sigma_z(wave_function, configurations):
    total_spins_per_sample = numpy.prod(configurations.shape[1:])
    axis_to_sum = tuple((range(1, len(configurations.shape))))
    return configurations.sum(axis=axis_to_sum) / total_spins_per_sample


AbsSigmaZ = functools.partial(LambdaObservable, observable_function=abs_sigma_z)
SigmaZ = functools.partial(LambdaObservable, observable_function=sigma_z)
