import functools

from ..deepar.samplers import Sampler, FastAutoregressiveSampler, \
    AutoregressiveSampler as ZeroBasedAutoregressiveSampler
from .exact_sampler import ExactSampler
from .ensemble import Ensemble
from .metropolis_hastings import MetropolisHastingsSampler, MetropolisHastingsLocal, MetropolisHastingsUniform, \
    MetropolisHastingsHamiltonian

AutoregressiveSampler = functools.partial(ZeroBasedAutoregressiveSampler, zero_base=False)
