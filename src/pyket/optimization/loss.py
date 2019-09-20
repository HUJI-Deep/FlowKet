import tensorflow


def loss_for_energy_minimization(y_true, y_pred):
    return 2.0 * tensorflow.math.real(tensorflow.math.multiply(y_pred, y_true))
