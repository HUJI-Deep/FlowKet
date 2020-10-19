import tensorflow


class SimpleClass(object):
    pass


def fix_tensorflow_v1_names():
    tensorflow.math.real = tensorflow.real
    tensorflow.math.imag = tensorflow.imag
    tensorflow.math.conj = tensorflow.conj
    tensorflow.math.log = tensorflow.log
    tensorflow.math.tanh = tensorflow.tanh
    tensorflow.math.reduce_logsumexp = tensorflow.reduce_logsumexp
    tensorflow.math.reduce_sum = tensorflow.reduce_sum
    tensorflow.math.reduce_mean = tensorflow.reduce_mean 
    tensorflow.math.reduce_max = tensorflow.reduce_max
    tensorflow.math.reduce_min = tensorflow.reduce_min
    tensorflow.math.multiply = tensorflow.multiply
    tensorflow.math.sqrt = tensorflow.sqrt
    tensorflow.math.floormod = tensorflow.floormod
    tensorflow.math.abs = tensorflow.abs
    tensorflow.random = SimpleClass()
    tensorflow.random.normal = tensorflow.random_normal 
    tensorflow.random.uniform = tensorflow.random_uniform
    def categorical_func(logits, num_samples, dtype=None, seed=None, name=None):
        return tensorflow.multinomial(logits, num_samples, output_dtype=dtype, seed=seed, name=name)
    tensorflow.random.categorical = categorical_func
    if not hasattr(tensorflow, 'roll'):
        tensorflow.roll = tensorflow.manip.roll
