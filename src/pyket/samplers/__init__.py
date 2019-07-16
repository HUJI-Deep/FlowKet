from ..deepar.samplers import Sampler, Ensemble
from .exact_sampler import ExactSampler, WaveFunctionSampler
from .fast_autoregressive import FastAutoregressiveSampler
from .metropolis_hastings import MetropolisHastingsSampler, MetropolisHastingsLocal, MetropolisHastingsUniform, \
    MetropolisHastingsHamiltonian, MetropolisHastingsExchange


def _build_autoregressive_sampler():
    import functools
    from ..deepar.samplers import AutoregressiveSampler as ZeroBasedAutoregressiveSampler
    return functools.partial(ZeroBasedAutoregressiveSampler, zero_base=False)


AutoregressiveSampler = _build_autoregressive_sampler()
