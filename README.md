# FlowKet - A Python framework for variational Monte-Carlo simulations on top of Tensorflow

FlowKet is our framework for running variational Monte-Carlo simulations of quantum many-body systems. It supports any Keras model for representing a parameterized unnormalized wave-function, e.g., Restricted Boltzman Machines and ConvNets, with real or complex-valued parameters. We have implemented a standard Markov-Chain Monte-Carlo (MCMC) energy gradient estimator for this general case, which can be used to approximate the ground state of a quantum system according to a given Hamiltonian. The neural-network-based approach for representing wave-fucntions was shown to be a promising method for solving the many-body problem, often matching or even surpassing the precision of other competing methods.

In addition to an MCMC energy gradient estimator, we have also implemented our novel Neural Autoregressive Quantum State wave-function representation that supports efficient and exact sampling. By overcoming the reliance on MCMC, our models can converge much faster for models of same size, which allows us to scale them to millions of parameters, as opposed to just a few thousands for prior approaches. This leads to better precison and ability to invesitgate larger and more intricated systems. Please [read our paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.020503) ([arXiv version](https://arxiv.org/abs/1902.04057)), cited below, for further details on this approach. We hope that users of our library will be able to take our method and apply to a variety of problems. If you use this codebase or apply our method, we would appreciate if you cite us as follows:
```bibtex
@article{PhysRevLett.124.020503,
  title = {Deep Autoregressive Models for the Efficient Variational Simulation of Many-Body Quantum Systems},
  author = {Sharir, Or and Levine, Yoav and Wies, Noam and Carleo, Giuseppe and Shashua, Amnon},
  journal = {Phys. Rev. Lett.},
  volume = {124},
  issue = {2},
  pages = {020503},
  numpages = {6},
  year = {2020},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.124.020503},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.124.020503}
}
```

## Installation

FlowKet assumes Tensorflow is already part of the enviornment. We currently support Tensorflow 1.10-1.14, but plan to extend support to all >=1.10+ and 2.0.

The recommended way to intall FlowKet is via PyPI:
```bash
pip install flowket
```

Alternatively if you wish to work on extending our library, you can clone our project and instead run:
```bash
pip install -e /path/to/local/repo
```

## Basic Tutorial

While we are working on writing a proper tutorial on using the framework, we suggest going through the example files.
