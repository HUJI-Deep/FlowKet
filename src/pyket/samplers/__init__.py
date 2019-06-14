import functools

from ..deepar.samplers import Sampler, Ensemble, AutoregressiveSampler as ZeroBasedAutoregressiveSampler
from .exact_sampler import ExactSampler
from .fast_autoregressive import FastAutoregressiveSampler
from .metropolis_hastings import MetropolisHastingsSampler, MetropolisHastingsLocal, MetropolisHastingsUniform, \
    MetropolisHastingsHamiltonian

AutoregressiveSampler = functools.partial(ZeroBasedAutoregressiveSampler, zero_base=False)
