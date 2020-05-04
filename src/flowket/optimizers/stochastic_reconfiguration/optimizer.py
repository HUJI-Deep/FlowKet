from collections import namedtuple

import tensorflow as tf
from tensorflow.python.keras import backend as K

from .linear_equations import conjugate_gradient
from ..complex_values_optimizer import ComplexValuesOptimizer
from ..utils import tensors_to_column
from ...layers.complex.tensorflow_ops import float_norm

Operator = namedtuple('Operator', 'shape,dtype,apply')


class ComplexValuesStochasticReconfiguration(ComplexValuesOptimizer):
    """docstring for StochasticReconfiguration"""

    def __init__(self, predictions_keras_model, predictions_jacobian, lr=0.01, diag_shift=0.05,
                 iterative_solver=True, compute_jvp_instead_of_full_jacobian=False,
                 use_energy_loss=False, add_s_matrix_stats=False, conjugate_gradient_tol=1e-3,
                 iterative_solver_max_iterations=200, use_cholesky=True, **kwargs):
        super(ComplexValuesStochasticReconfiguration, self).__init__(predictions_keras_model,
                                                                     predictions_jacobian, lr=lr, **kwargs)
        self.add_s_matrix_stats = add_s_matrix_stats
        self.compute_jvp_instead_of_full_jacobian = compute_jvp_instead_of_full_jacobian
        self.conjugate_gradient_tol = conjugate_gradient_tol
        self.use_energy_loss = use_energy_loss
        self.use_cholesky = use_cholesky
        self.iterative_solver = iterative_solver
        self.iterative_solver_max_iterations = iterative_solver_max_iterations
        self._compute_batch_size()
        self._init_optimizer_parameters(diag_shift, lr)

    def get_updates(self, loss, params):
        assert params == self.predictions_keras_model.weights
        wave_function_jacobian_minus_mean = None
        if not (self.iterative_solver and self.compute_jvp_instead_of_full_jacobian):
            wave_function_jacobian_minus_mean = self.get_wave_function_jacobian_minus_mean()
        energy_grad = self.get_energy_grad(loss, wave_function_jacobian_minus_mean)
        flat_gradient = self.compute_wave_function_gradient_covariance_inverse_multiplication(
            energy_grad, wave_function_jacobian_minus_mean)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates += self.apply_complex_gradient(flat_gradient * (-1.0 + 0j))
        return self.updates

    def compute_wave_function_gradient_covariance_inverse_multiplication(self, complex_vector,
                                                                         wave_function_jacobian_minus_mean):
        if self.iterative_solver:
            res = self.compute_wave_function_gradient_covariance_inverse_multiplication_with_iterative_solver(
                complex_vector, wave_function_jacobian_minus_mean)
        else:
            res = self.compute_wave_function_gradient_covariance_inverse_multiplication_directly(
                complex_vector, wave_function_jacobian_minus_mean)
        return res

    def compute_wave_function_gradient_covariance_inverse_multiplication_directly(self, complex_vector,
                                                                                  wave_function_jacobian_minus_mean):
        num_of_complex_params_t = tf.shape(complex_vector)[0]
        s = tf.matmul(wave_function_jacobian_minus_mean, wave_function_jacobian_minus_mean,
                      adjoint_a=True) / self.batch_size
        updated_s_matrix_min_eigval, updated_s_matrix_rank = self._update_s_matrix_stats(num_of_complex_params_t, s)
        s += tf.eye(num_of_complex_params_t, dtype=self.predictions_keras_model.output.dtype) * self.diag_shift
        with tf.control_dependencies([updated_s_matrix_min_eigval, updated_s_matrix_rank]):
            if self.use_cholesky:
                res = tf.linalg.cholesky_solve(tf.linalg.cholesky(s), complex_vector)
            else:
                res = tf.linalg.solve(s, complex_vector)
            return tf.stop_gradient(res)

    def compute_wave_function_gradient_covariance_inverse_multiplication_with_iterative_solver(self,
                                                                                               complex_vector,
                                                                                               wave_function_jacobian_minus_mean=None):
        complex_vector = tf.squeeze(complex_vector)
        num_of_complex_params_t = tf.shape(complex_vector)[:1]
        if wave_function_jacobian_minus_mean is None:
            def wave_function_gradient_covariance_vector_product(complex_vector):
                return self.get_stochastic_reconfiguration_matrix_vector_product_via_jvp(complex_vector)
        else:
            def wave_function_gradient_covariance_vector_product(v):
                return tf.matmul(wave_function_jacobian_minus_mean,
                                 tf.matmul(wave_function_jacobian_minus_mean, v),
                                 adjoint_a=True) / self.batch_size + self.diag_shift * v
        operator = Operator(shape=tf.concat([num_of_complex_params_t] * 2, axis=0),
                            dtype=self.predictions_keras_model.output.dtype,
                            apply=wave_function_gradient_covariance_vector_product)
        conjugate_gradient_res = conjugate_gradient(operator, complex_vector, tol=self.conjugate_gradient_tol,
                                                    max_iter=self.iterative_solver_max_iterations)
        updated_conjugate_gradient_iterations = K.update(self.conjugate_gradient_iterations, conjugate_gradient_res.i)
        updated_conjugate_gradient_residual_norm = K.update(self.conjugate_gradient_residual_norm,
                                                            float_norm(conjugate_gradient_res.r))
        with tf.control_dependencies([updated_conjugate_gradient_iterations, updated_conjugate_gradient_residual_norm]):
            flat_gradient = tf.stop_gradient(tf.reshape(conjugate_gradient_res.x, (-1, 1)))
        return flat_gradient

    def get_energy_grad(self, loss, wave_function_jacobian_minus_mean=None):
        if self.use_energy_loss:
            # todo fix this branch!
            energy_grads = self.get_model_parameters_complex_value_gradients(loss)
            # we take conjugate because our loss actually calculate the conj gradient and usually it's ok because just
            # take the real part ...
            energy_grad = tf.conj(tensors_to_column(energy_grads)) / 2
        else:
            complex_vector = tf.conj(tf.reshape(self.predictions_keras_model.targets[0],
                                                (-1, 1)))
            if wave_function_jacobian_minus_mean is None:
                energy_grad = self.get_predictions_jacobian_vector_product(complex_vector,
                                                                           conjugate_jacobian=True)
            else:
                energy_grad = tf.matmul(wave_function_jacobian_minus_mean, complex_vector, adjoint_a=True)
        return energy_grad

    def get_wave_function_jacobian_minus_mean(self):
        jacobian_complex = self.get_predictions_jacobian()
        mean_grad = tf.math.reduce_mean(jacobian_complex, axis=0, keepdims=True)
        return jacobian_complex - mean_grad

    def get_stochastic_reconfiguration_matrix_vector_product_via_jvp(self, complex_vector):
        mean_pred = tf.math.reduce_mean(tf.math.real(self.predictions_keras_model.output))
        mean_grad = tensors_to_column(self.get_model_parameters_complex_value_gradients(mean_pred))
        jvp = self.get_predictions_jacobian_vector_product(complex_vector, conjugate_jacobian=True)
        ok_remainder = tf.squeeze(tf.matmul(tensors_to_column(complex_vector), mean_grad, transpose_a=True))
        ok_v = jvp - ok_remainder
        vjp = self.get_model_parameters_complex_value_gradients(self.predictions_keras_model.output,
                                                                grad_ys=ok_v, conjugate_gradients=True)
        return tensors_to_column(vjp) / self.batch_size + complex_vector * self.diag_shift

    @property
    def metrics(self):
        def conjugate_gradient_residual_norm(y_true, y_pred):
            return self.conjugate_gradient_residual_norm

        def conjugate_gradient_iterations(y_true, y_pred):
            return self.conjugate_gradient_iterations

        def s_matrix_min_eigval(y_true, y_pred):
            return self.s_matrix_min_eigval

        def s_matrix_rank(y_true, y_pred):
            return self.s_matrix_rank

        conjugate_gradient_residual_norm.__name__ = 'stochastic_reconfiguration/%s' % conjugate_gradient_residual_norm.__name__
        conjugate_gradient_iterations.__name__ = 'stochastic_reconfiguration/%s' % conjugate_gradient_iterations.__name__
        s_matrix_min_eigval.__name__ = 'stochastic_reconfiguration/%s' % s_matrix_min_eigval.__name__
        s_matrix_rank.__name__ = 'stochastic_reconfiguration/%s' % s_matrix_rank.__name__

        res = []
        if self.iterative_solver:
            res.append(conjugate_gradient_iterations)
            res.append(conjugate_gradient_residual_norm)
        elif self.add_s_matrix_stats:
            res.append(s_matrix_rank)
            res.append(s_matrix_min_eigval)
        return res

    def _compute_batch_size(self):
        self.batch_size = tf.cast(tf.shape(self.predictions_keras_model.output)[0],
                                  self.predictions_keras_model.output.dtype)

    def _init_optimizer_parameters(self, diag_shift, lr):
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.diag_shift = K.variable(diag_shift, name='diag_shift', dtype=self.predictions_keras_model.output.dtype)
            self.s_matrix_min_eigval = K.variable(0.0, name='s_matrix_min_eigval',
                                                  dtype=self.predictions_keras_model.output.dtype.real_dtype)
            self.conjugate_gradient_iterations = K.variable(0, name='conjugate_gradient_iterations', dtype='int32')
            self.conjugate_gradient_residual_norm = K.variable(0.0, name='conjugate_gradient_residual_norm',
                                                               dtype='float64')
            self.s_matrix_rank = K.variable(0, dtype='int64', name='s_matrix_rank')
        if tf.__version__ >= '1.14':
            self._weights += [self.iterations, self.diag_shift]
        else:
            self.weights = [self.iterations, self.diag_shift]

    def _update_s_matrix_stats(self, num_of_complex_params_t, s):
        if not self.add_s_matrix_stats:
            return tf.no_op(), tf.no_op()
        abs_eigvals = tf.math.abs(tf.linalg.eigvalsh(s))
        tol = K.epsilon() * tf.cast(num_of_complex_params_t, abs_eigvals.dtype) * tf.math.reduce_max(
            tf.math.abs(s))  # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.matrix_rank.html
        filtered_eigvals = tf.boolean_mask(abs_eigvals, abs_eigvals > tol)
        updated_s_matrix_rank = K.update(self.s_matrix_rank, tf.count_nonzero(filtered_eigvals))
        updated_s_matrix_min_eigval = K.update(self.s_matrix_min_eigval, tf.math.reduce_min(filtered_eigvals))
        return updated_s_matrix_min_eigval, updated_s_matrix_rank
