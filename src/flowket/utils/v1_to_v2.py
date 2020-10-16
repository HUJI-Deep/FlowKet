import tensorflow

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
    tensorflow.random.normal = tensorflow.random_normal 
    tensorflow.random.uniform = tensorflow.random_uniform
    if not hasattr(tensorflow, 'roll'):
        tensorflow.roll = tensorflow.manip.roll
