from .autoregressive import AutoregressiveSampler as ZerosBaseAutoregressiveSampler
from .base_sampler import Sampler
from .exact_sampler import ExactSampler
from .ensemble import Ensemble
from .fast_autoregressive import FastAutoregressiveSampler as ZerosBaseFastAutoregressiveSampler
from .metropolis_hastings import MetropolisHastingsSampler, MetropolisHastingsLocal, MetropolisHastingsUniform, \
    MetropolisHastingsHamiltonian
from .to_plus_minus_one_decorator import ToPlusMinusOneDecorator


def autoregressive_sampler(conditional_log_probs_machine, batch_size,
                           **kwargs):
    sampler = ZerosBaseAutoregressiveSampler(conditional_log_probs_machine,
                                             batch_size,
                                             **kwargs)
    return ToPlusMinusOneDecorator(sampler)


def fast_autoregressive_sampler(conditional_log_probs_machine, batch_size,
                                **kwargs):
    sampler = ZerosBaseFastAutoregressiveSampler(conditional_log_probs_machine,
                                                 batch_size,
                                                 **kwargs)
    return ToPlusMinusOneDecorator(sampler)


AutoregressiveSampler = autoregressive_sampler
FastAutoregressiveSampler = fast_autoregressive_sampler
