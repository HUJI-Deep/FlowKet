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
    model_weights = [layer.weights_for_complex_value_params_conj_gradient for layer in keras_model.layers
                     if layer.count_params() > 0]
    return [item for sublist in model_weights for item in sublist]


def get_model_real_weights(keras_model):
    model_weights = [layer.real_weights for layer in keras_model.layers if layer.count_params() > 0]
    return [item for sublist in model_weights for item in sublist]


def get_model_imag_weights(keras_model):
    model_weights = [layer.imag_weights for layer in keras_model.layers if layer.count_params() > 0]
    return [item for sublist in model_weights for item in sublist]


class ComplexValuesStochasticReconfiguration(Optimizer):
    """docstring for StochasticReconfiguration"""

    def __init__(self, predictions_keras_model, predictions_jacobian, lr=0.01, diag_shift=0.05, iterative_solver=True,
                 compute_jvp_instead_of_full_jacobian=False, conjugate_gradient_tol=1e-3, max_iter=200, plain_local_energy_loss=False, **kwargs):
        super(ComplexValuesStochasticReconfiguration, self).__init__(**kwargs)
        self.predictions_keras_model = predictions_keras_model
        self.predictions_jacobian = predictions_jacobian
        self.iterative_solver = iterative_solver
        self.compute_jvp_instead_of_full_jacobian = compute_jvp_instead_of_full_jacobian
        self.conjugate_gradient_tol = conjugate_gradient_tol
        if plain_local_energy_loss:
            assert not (self.iterative_solver and compute_jvp_instead_of_full_jacobian)
        self.plain_local_energy_loss = plain_local_energy_loss
        self.max_iter = max_iter
        self.model_weights_for_complex_value_params_conj_gradient = get_model_weights_for_complex_value_params_gradient(
            self.predictions_keras_model)
        self.model_real_weights = get_model_real_weights(self.predictions_keras_model)
        self.model_imag_weights = get_model_imag_weights(self.predictions_keras_model)
        self.batch_size = tf.cast(tf.shape(self.predictions_keras_model.output)[0],
                                  self.predictions_keras_model.output.dtype)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.diag_shift = K.variable(diag_shift, name='diag_shift', dtype=self.predictions_keras_model.output.dtype)

    def get_updates(self, loss, params):
        assert params == self.predictions_keras_model.weights
        if self.iterative_solver and self.compute_jvp_instead_of_full_jacobian:
            wave_function_jacobian_minus_mean = None
        else:
            wave_function_jacobian_minus_mean = self.get_wave_function_jacobian_minus_mean()
        if self.plain_local_energy_loss:
            energy_grad = tf.matmul(wave_function_jacobian_minus_mean, tf.conj(tf.reshape(self.predictions_keras_model.targets[0], (-1, 1))), adjoint_a=True)
        else:
            energy_grad = self.get_energy_grad(loss)
        if self.iterative_solver:
            flat_gradient = self.compute_wave_function_gradient_covariance_inverse_multiplication_with_iterative_solver(
                energy_grad, wave_function_jacobian_minus_mean)
        else:
            flat_gradient = self.compute_wave_function_gradient_covariance_inverse_multiplication_directly(
                energy_grad, wave_function_jacobian_minus_mean)

        return self.apply_complex_gradient(flat_gradient * (-1.0 + 0j))

    def apply_complex_gradient(self, flat_gradient):
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

    def compute_wave_function_gradient_covariance_inverse_multiplication_directly(self, complex_vector,
                                                                                  wave_function_jacobian_minus_mean):
        num_of_complex_params_t = tf.shape(complex_vector)[0]
        s = tf.matmul(wave_function_jacobian_minus_mean, wave_function_jacobian_minus_mean,
                      adjoint_a=True) / self.batch_size
        s += tf.eye(num_of_complex_params_t, dtype=self.predictions_keras_model.output.dtype) * self.diag_shift
        return tf.stop_gradient(tf.linalg.solve(s, complex_vector))

    def compute_wave_function_gradient_covariance_inverse_multiplication_with_iterative_solver(self,
                                                                                               complex_vector,
                                                                                               wave_function_jacobian_minus_mean=None):
        complex_vector = tf.squeeze(complex_vector)
        num_of_complex_params_t = tf.shape(complex_vector)[0]
        if wave_function_jacobian_minus_mean is None:
            def wave_function_gradient_covariance_vector_product(complex_vector):
                return self.get_stochastic_reconfiguration_matrix_vector_product_via_jvp(complex_vector)
        else:
            def wave_function_gradient_covariance_vector_product(v):
                return tf.matmul(wave_function_jacobian_minus_mean,
                                 tf.matmul(wave_function_jacobian_minus_mean, v),
                                 adjoint_a=True) / self.batch_size + self.diag_shift * v
        operator = Operator(shape=tf.stack([num_of_complex_params_t] * 2, axis=0),
                            dtype=self.predictions_keras_model.output.dtype,
                            apply=wave_function_gradient_covariance_vector_product)
        conjugate_gradient_res = conjugate_gradient(operator, complex_vector, tol=self.conjugate_gradient_tol,
                                                    max_iter=self.max_iter)
        flat_gradient = tf.stop_gradient(conjugate_gradient_res.x)
        with tf.name_scope(None, "StochasticReconfiguration", []) as name:
            tf.summary.scalar('conjugate_gradient_iterations', conjugate_gradient_res.i)
            tf.summary.scalar('conjugate_gradient_residual_norm', float_norm(conjugate_gradient_res.r))
        return flat_gradient

    def get_energy_grad(self, loss):
        energy_grads = self.get_complex_value_gradients(loss)
        # we take conjugate because our loss actually calculate the conj gradient and usually it's ok because just
        # take the real part ...
        return tf.conj(tensors_to_column(energy_grads)) / 2

    def get_complex_value_gradients(self, loss):
        return to_complex_tensors(self.get_gradients(loss, self.predictions_keras_model.weights))

    def get_wave_function_jacobian_minus_mean(self):
        jacobian_complex = self.get_wave_function_jacobian()
        mean_grad = tf.reduce_mean(jacobian_complex, axis=0, keepdims=True)
        return jacobian_complex - mean_grad

    def get_wave_function_jacobian(self):
        jacobian_real = self.predictions_jacobian(self.predictions_keras_model.weights)
        return tensors_to_matrix(to_complex_tensors(jacobian_real),
                                             tf.shape(jacobian_real[0])[0])

    def _from_complex_vector(self, complex_vector):
        return tensors_to_column([tf.concat([r, i], axis=0) for r, i in
                                  zip(column_to_tensors(self.model_real_weights, tf.real(complex_vector)),
                                      column_to_tensors(self.model_real_weights, tf.imag(complex_vector)))])

    def get_stochastic_reconfiguration_matrix_vector_product_via_jvp(self, complex_vector):
        vector = self._from_complex_vector(complex_vector)
        output_tensor = self.predictions_keras_model.output
        mean_grads = tf.gradients(tf.reduce_mean(tf.real(output_tensor)), self.predictions_keras_model.weights)
        mean_grad = tensors_to_column(to_complex_tensors(mean_grads))
        jvp = forward_mode_gradients([output_tensor], self.model_weights_for_complex_value_params_conj_gradient,
                                     column_to_tensors(self.model_weights_for_complex_value_params_conj_gradient, vector))[0]
        ok_remainder = tf.squeeze(tf.matmul(tensors_to_column(
            complex_vector), mean_grad, transpose_a=True))
        ok_v = jvp - ok_remainder
        vjp = tf.gradients(output_tensor, self.model_weights_for_complex_value_params_conj_gradient, grad_ys=ok_v,
                           colocate_gradients_with_ops=True)
        # ok_t_remainder is zeros !!!
        return tensors_to_column(to_complex_tensors(vjp)) / self.batch_size + complex_vector * self.diag_shift
