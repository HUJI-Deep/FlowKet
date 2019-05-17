from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K

from .linear_equations import conjugate_gradient
from .utils import tensors_to_column, column_to_tensors, \
    forward_mode_gradients, tensors_to_matrix
from ...layers.complex.tensorflow_ops import float_norm

Operator = namedtuple('Operator', 'shape,dtype,apply')


def to_complex_tensors(tensors):
    return [tf.complex(r, i) for r, i in zip(*[iter(tensors)] * 2)]


def get_model_weights_for_complex_value_params_gradient(keras_model):
    model_weights = [layer.weights_for_complex_value_params_gradient for layer in keras_model.layers]
    return [item for sublist in model_weights for item in sublist]


def get_model_real_weights(keras_model):
    model_weights = [layer.real_weights for layer in keras_model.layers]
    return [item for sublist in model_weights for item in sublist]


def get_model_imag_weights(keras_model):
    model_weights = [layer.imag_weights for layer in keras_model.layers]
    return [item for sublist in model_weights for item in sublist]


class ComplexValueParametersStochasticReconfiguration(Optimizer):
    """docstring for StochasticReconfiguration"""

    def __init__(self, predictions_keras_model, predictions_jacobian, lr=0.01, diag_shift=0.05, iterative_solver=True,
                 use_fast_sr=False, conjugate_gradient_tol=1e-3, max_iter=200, **kwargs):
        super(ComplexValueParametersStochasticReconfiguration, self).__init__(**kwargs)
        self.predictions_keras_model = predictions_keras_model
        self.predictions_jacobian = predictions_jacobian
        self.iterative_solver = iterative_solver
        self.use_fast_sr = use_fast_sr
        self.conjugate_gradient_tol = conjugate_gradient_tol
        self.max_iter = max_iter
        self.model_weights_for_complex_value_params_gradient = get_model_weights_for_complex_value_params_gradient(
            self.predictions_keras_model)
        self.model_real_weights = get_model_real_weights(self.predictions_keras_model)
        self.model_imag_weights = get_model_imag_weights(self.predictions_keras_model)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.diag_shift = K.variable(diag_shift, name='diag_shift')

    def from_complex_vector(self, complex_vector):
        return tensors_to_column([tf.concat([r, i], axis=0) for r, i in
                                  zip(column_to_tensors(self.model_real_weights, tf.real(complex_vector)),
                                      column_to_tensors(self.model_real_weights, tf.imag(complex_vector)))])

    def get_fast_stochastic_reconfiguration_matrix_vector_product(self, complex_vector):
        vector = self.from_complex_vector(complex_vector)
        output_tensor = self.predictions_keras_model.output
        mean_grads = tf.gradients(tf.reduce_mean(tf.real(output_tensor)), self.predictions_keras_model.weights)
        mean_grad = tensors_to_column(to_complex_tensors(mean_grads))
        jvp = forward_mode_gradients([output_tensor], self.model_weights_for_complex_value_params_gradient,
                                     column_to_tensors(self.model_weights_for_complex_value_params_gradient, vector))[0]
        ok_remainder = tf.squeeze(tf.matmul(tensors_to_column(
            complex_vector), mean_grad, transpose_a=True))
        ok_v = jvp - ok_remainder
        vjp = tf.gradients(output_tensor, self.model_weights_for_complex_value_params_gradient, grad_ys=ok_v,
                           colocate_gradients_with_ops=True)
        # ok_t_remainder is zeros !!!
        return tensors_to_column(to_complex_tensors(vjp)) / tf.cast(tf.shape(jvp)[0],
                                                                    self.predictions.dtype) + complex_vector * self.diag_shift

    def get_updates(self, loss, params):
        assert params == self.predictions_keras_model.weights
        energy_grads = self.get_gradients(loss, self.model_weights_for_complex_gradient)
        energy_grad = tf.squeeze(tensors_to_column(
            to_complex_tensors(energy_grads))) / 2
        num_of_params_t = tf.shape(energy_grad)[0]
        op_shape_t = tf.stack([num_of_params_t] * 2, axis=0)
        jacobian_real = self.predictions_jacobian(params)
        jacobian_complex = tensors_to_matrix(to_complex_tensors(jacobian_real),
                                             tf.shape(jacobian_real)[0])
        mean_grad = tf.reduce_mean(jacobian_complex, axis=0, keepdims=True)
        ok = jacobian_complex - mean_grad

        batch_size = tf.cast(tf.shape(jacobian_real)[0], self.predictions.dtype)

        if not self.iterative_solver:
            s = tf.matmul(ok, ok, adjoint_a=True) / batch_size \
                + tf.eye(num_of_params_t, dtype=self.predictions.dtype) * self.diag_shift
            flat_gradient = tf.linalg.solve(s, tensors_to_column(to_complex_tensors(energy_grads)))
        else:
            if self.use_fast_sr:
                def sr_vector_product(complex_vector):
                    return self.get_fast_stochastic_reconfiguration_matrix_vector_product(complex_vector)
            else:
                def sr_vector_product(complex_vector):
                    return tf.matmul(ok, tf.matmul(ok, complex_vector),
                                     adjoint_a=True) / batch_size + self.diag_shift * complex_vector

            operator = Operator(shape=op_shape_t, dtype=self.predictions.dtype,
                                apply=sr_vector_product)
            conjugate_gradient_res = conjugate_gradient(operator, energy_grad, tol=self.conjugate_gradient_tol,
                                                        max_iter=self.max_iter)
            flat_gradient = tf.stop_gradient(conjugate_gradient_res.x)
            with tf.name_scope(None, "StochasticReconfiguration", []) as name:
                tf.summary.scalar('conjugate_gradient_iterations', conjugate_gradient_res.i)
                tf.summary.scalar('conjugate_gradient_residual_norm', float_norm(conjugate_gradient_res.r))
        conj_flat_gradient = tf.conj(flat_gradient)
        real_gradients = column_to_tensors(self.model_real_weights, tf.real(conj_flat_gradient))
        imag_gradients = column_to_tensors(self.model_imag_weights, tf.imag(conj_flat_gradient))
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations]
        for p, g in zip(self.model_real_weights + self.model_imag_weights, real_gradients + imag_gradients):
            new_p = p + self.lr * g
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates
