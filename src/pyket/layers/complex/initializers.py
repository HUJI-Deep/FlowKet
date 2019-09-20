import abc
import math

import tensorflow
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.initializers import get as get_real_number_initializer
from tensorflow.python.ops.init_ops import _compute_fans


class ComplexValueInitializer(abc.ABC):
    """docstring for ComplexNumberInitializer"""

    def __init__(self):
        super(ComplexValueInitializer, self).__init__()

    @abc.abstractmethod
    def get_real_part_initializer(self):
        pass

    @abc.abstractmethod
    def get_imag_part_initializer(self):
        pass


class NegateDecorator(Initializer):
    """docstring for NegateDecorator"""

    def __init__(self, initializer):
        super(NegateDecorator, self).__init__()
        self.initializer = initializer

    def __call__(self, shape, dtype=None, partition_info=None):
        return -1 * self.initializer.__call__(shape, dtype, partition_info)


class ConjugateDecorator(ComplexValueInitializer):
    """docstring for ConjugateDecorator"""

    def __init__(self, complex_initializer):
        super(ConjugateDecorator, self).__init__()
        self.complex_initializer = complex_initializer

    def get_real_part_initializer(self):
        return self.complex_initializer.get_real_part_initializer()

    def get_imag_part_initializer(self):
        return NegateDecorator(self.complex_initializer.get_imag_part_initializer())


class FromRealValueInitializers(ComplexValueInitializer):
    """docstring for FromRealValueInitializers"""

    def get_real_part_initializer(self):
        return self.real_part_initializer

    def get_imag_part_initializer(self):
        return self.imag_part_initializer

    def __init__(self, real_part_initializer, imag_part_initializer):
        super(FromRealValueInitializers, self).__init__()
        self.real_part_initializer = get_real_number_initializer(real_part_initializer)
        self.imag_part_initializer = get_real_number_initializer(imag_part_initializer)


class _RealPartInitializer(Initializer):
    """docstring for _RealPartInitializer"""

    def __init__(self, whole_initializer):
        super(_RealPartInitializer, self).__init__()
        self.whole_initializer = whole_initializer

    def __call__(self, shape, dtype=None, partition_info=None):
        self.whole_initializer.next_complex_nummber(shape, dtype=dtype)
        return tensorflow.math.cos(self.whole_initializer.phase) * self.whole_initializer.modulus


class _ImagPartInitializer(Initializer):
    """docstring for _RealPartInitializer"""

    def __init__(self, whole_initializer):
        super(_ImagPartInitializer, self).__init__()
        self.whole_initializer = whole_initializer

    def __call__(self, shape, dtype=None, partition_info=None):
        self.whole_initializer.next_complex_nummber(shape, dtype=dtype)
        return tensorflow.math.sin(self.whole_initializer.phase) * self.whole_initializer.modulus


def random_rayleigh(shape, scale):
    scale_squared = scale * scale
    x = tensorflow.random_normal(shape, stddev=scale_squared)
    y = tensorflow.random_normal(shape, stddev=scale_squared)
    return tensorflow.math.sqrt(x * x + y * y)


def to_int_shape(shape):
    return tuple([int(s) for s in shape])


class StandartComplexValueInitializer(ComplexValueInitializer):
    """docstring for StandartComplexNumberInitializer"""

    def get_real_part_initializer(self):
        return self.real_part_initializer

    def get_imag_part_initializer(self):
        return self.imag_part_initializer

    def _random_modulus_and_phase(self, shape):
        shape = to_int_shape(shape)
        fan_in, fan_out = _compute_fans(shape)
        if self.criterion == 'glorot':
            scale = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            scale = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        self.modulus = random_rayleigh(shape, scale=scale)
        self.phase = tensorflow.random_uniform(shape, minval=-math.pi, maxval=math.pi)

    def next_complex_nummber(self, shape, dtype=None):
        self._counter += 1
        if self._counter == 2:
            return
        if self._counter > 2:
            raise Exception('You should create unique instance of this initializer for each complex value parameter')
        self._random_modulus_and_phase(shape)

    def __init__(self, criterion='glorot'):
        super(StandartComplexValueInitializer, self).__init__()
        self.criterion = criterion
        self._counter = 0
        self.real_part_initializer = _RealPartInitializer(self)
        self.imag_part_initializer = _ImagPartInitializer(self)


def get(identifier):
    if isinstance(identifier, ComplexValueInitializer):
        return identifier
    elif isinstance(identifier, tuple) and len(identifier) == 2:
        return FromRealValueInitializers(identifier[0], identifier[1])
    elif identifier in ['complex_glorot', 'complex_he']:
        return StandartComplexValueInitializer(identifier[8:])
    else:
        return FromRealValueInitializers(identifier, identifier)
