import tensorflow


def energy_gradient_loss(y_true, y_pred):
    return 2.0 * tensorflow.real(tensorflow.multiply(y_pred, y_true))


def monte_carlo_generator_return_local_energy_minus_mean_loss(y_true, y_pred):
    return tensorflow.conj(y_true)
