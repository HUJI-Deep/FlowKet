import pytest
import tensorflow as tf


from pyket.callbacks.exact import ExactLocalEnergy
from pyket.callbacks.monte_carlo import LocalEnergyStats
from pyket.evaluation import evaluate, exact_evaluate
from pyket.operators import Heisenberg
from pyket.optimization import ExactVariational, VariationalMonteCarlo
from pyket.samplers import ExactSampler
from .simple_models import complex_values_linear_1d_model, real_values_1d_model


DEFAULT_TF_GRAPH = tf.get_default_graph()
ONE_DIM_OPERATOR = Heisenberg(hilbert_state_shape=[7], pbc=True)


@pytest.mark.parametrize('model_builder, operator, batch_size', [
    (real_values_1d_model, ONE_DIM_OPERATOR , 2 ** 10),
    (complex_values_linear_1d_model, ONE_DIM_OPERATOR , 2 ** 10),
])
def test_exact_and_monte_carlo_agree(model_builder, operator, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        model = model_builder()
        exact_variational = ExactVariational(model, operator, batch_size)
        sampler = ExactSampler(exact_variational, batch_size)
        variational_monte_carlo = VariationalMonteCarlo(model,operator, sampler)
        exact_energy = exact_evaluate(exact_variational,
                                      [ExactLocalEnergy(exact_variational)])['energy/energy']
        monte_carlo_energy = evaluate(variational_monte_carlo, 100,
                                      [LocalEnergyStats(variational_monte_carlo)])['energy/energy']
        assert monte_carlo_energy == pytest.approx(exact_energy)


def test_monte_carlo_and_netket_agree():
    pass