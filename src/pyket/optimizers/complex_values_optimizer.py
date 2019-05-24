import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K

from .utils import column_to_tensors, forward_mode_gradients, tensors_to_matrix, tensors_to_column


def to_complex_tensors(tensors):
    return [tf.complex(r, i) for r, i in zip(*[iter(tensors)] * 2)]


def get_model_weights_for_complex_value_params_gradient(keras_model):
    model_weights = [layer.weights_for_complex_value_params_gradient_conjugate for layer in keras_model.layers
                     if layer.count_params() > 0]
    return [item for sublist in model_weights for item in sublist]


def get_model_real_weights(keras_model):
    model_weights = [layer.real_weights for layer in keras_model.layers if layer.count_params() > 0]
    return [item for sublist in model_weights for item in sublist]


def get_model_imag_weights(keras_model):
    model_weights = [layer.imag_weights for layer in keras_model.layers if layer.count_params() > 0]
    return [item for sublist in model_weights for item in sublist]


class ComplexValuesOptimizer(Optimizer):
    """docstring for StochasticReconfiguration"""

    def __init__(self, predictions_keras_model, predictions_jacobian=None, **kwargs):
        super(ComplexValuesOptimizer, self).__init__(**kwargs)
        self.predictions_keras_model = predictions_keras_model
        self.predictions_jacobian = predictions_jacobian
        self.model_weights_for_complex_value_params_gradient_conjugate = get_model_weights_for_complex_value_params_gradient(
            self.predictions_keras_model)
        self.model_real_weights = get_model_real_weights(self.predictions_keras_model)
        self.model_imag_weights = get_model_imag_weights(self.predictions_keras_model)

    def apply_complex_gradient(self, flat_gradient):
        conj_flat_gradient = tf.conj(flat_gradient)
        real_gradients = column_to_tensors(self.model_real_weights, tf.real(conj_flat_gradient))
        imag_gradients = column_to_tensors(self.model_imag_weights, tf.imag(conj_flat_gradient))
        updates = []
        for p, g in zip(self.model_real_weights + self.model_imag_weights, real_gradients + imag_gradients):
            new_p = p + self.lr * g
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            updates.append(K.update(p, new_p))
        return updates

    def get_model_parameters_complex_value_gradients(self, loss, grad_ys=None, conjugate_gradients=False):
        params = self.model_weights_for_complex_value_params_gradient_conjugate \
            if conjugate_gradients else self.predictions_keras_model.weights
        return to_complex_tensors(tf.gradients(loss, params, grad_ys=grad_ys))

    def get_predictions_jacobian_vector_product(self, complex_vector, conjugate_jacobian=False):
        params = self.model_weights_for_complex_value_params_gradient_conjugate \
            if conjugate_jacobian else self.predictions_keras_model.weights
        grad_xs = self.from_complex_vector_to_tensors(complex_vector)
        grad_xs = column_to_tensors(params, tensors_to_column(grad_xs))
        return forward_mode_gradients([self.predictions_keras_model.output], params, grad_xs)[0]

    def get_predictions_jacobian(self):
        assert self.predictions_jacobian is not None
        jacobian_real = self.predictions_jacobian(self.predictions_keras_model.weights)
        return tensors_to_matrix(to_complex_tensors(jacobian_real),
                                 tf.shape(jacobian_real[0])[0])

    def from_complex_vector_to_tensors(self, complex_vector):

        return [tf.concat([r, i], axis=0) for r, i in
                zip(column_to_tensors(self.model_real_weights, tf.real(complex_vector)),
                    column_to_tensors(self.model_real_weights, tf.imag(complex_vector)))]
