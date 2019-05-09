import tensorflow

from tensorflow.keras import backend as K


def energy_gradient_loss(y_true, y_pred):
    # return 2.0 * K.sum(tensorflow.real(tensorflow.multiply(y_pred, y_true)))
    return 2.0 * tensorflow.real(tensorflow.multiply(y_pred, y_true))
