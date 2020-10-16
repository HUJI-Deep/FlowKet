import pytest
import tensorflow as tf

from flowket.layers.complex.tensorflow_ops import lncosh, float_norm

COMPLEX_NUMBERS = [2, 3j, 1 + 7j, 10 - 3j, -6]


@pytest.mark.parametrize('value', COMPLEX_NUMBERS)
def test_lncosh(value):
    if not tf.__version__.startswith('2'):
        tf.reset_default_graph()
    z = tf.constant([value], dtype=tf.complex128)
    our_res = lncosh(z)
    direct_calculation = tf.math.log(tf.cosh(z))
    diff_norm_t = float_norm(our_res - direct_calculation)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        diff_norm = sess.run(diff_norm_t)
        assert diff_norm.item() < 1e-8


@pytest.mark.parametrize('value', COMPLEX_NUMBERS)
def test_lncosh_gradient(value):
    if not tf.__version__.startswith('2'):
        tf.reset_default_graph()
    z = tf.constant([value], dtype=tf.complex128)
    our_res = lncosh(z)
    our_grad = tf.math.conj(tf.gradients(tf.math.real(our_res), z))
    real_grad = tf.math.tanh(z)
    diff_norm_t = float_norm(our_grad - real_grad)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        diff_norm = sess.run(diff_norm_t)
        assert diff_norm.item() < 1e-8
