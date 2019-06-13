import functools

from .fast_autoregressive import *
from ..deepar.samplers import Ensemble, Sampler, FastAutoregressiveSampler, AutoregressiveSampler as ZeroBasedAutoregressiveSampler
from .exact_sampler import ExactSampler
from .metropolis_hastings import MetropolisHastingsSampler, MetropolisHastingsLocal, MetropolisHastingsUniform, \
    MetropolisHastingsHamiltonian

AutoregressiveSampler = functools.partial(ZeroBasedAutoregressiveSampler, zero_base=False)