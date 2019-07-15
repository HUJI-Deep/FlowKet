import numpy

try:
    import accupy

    fdot = accupy.fdot


    def fsum(complex_array):
        if not complex_array.dtype.name.startswith('complex'):
            return accupy.fsum(complex_array)
        return accupy.fsum(numpy.real(complex_array)) + 1j * accupy.fsum(numpy.imag(complex_array))
except ImportError as e:
    fsum = numpy.sum
    fdot = numpy.dot


def decimal_to_binary(decimal, num_of_bits, zero_one_base=False):
    res = []
    for i in range(num_of_bits):
        if zero_one_base:
            res.append(decimal % 2)
        else:
            res.append(2 * (decimal % 2) - 1)
        decimal //= 2
    return res


def binary_to_decimal(binary_digits):
    res = 0
    power = 1
    for digit in binary_digits:
        if digit == 1:
            res += power
        power *= 2
    return res


def binary_array_to_decimal_array(binary_digits, out=None):
    if out is None:
        out = numpy.zeros((binary_digits.shape[:-1]), dtype='int32')
    power = 1
    for i in range(binary_digits.shape[-1]):
        out += (binary_digits[..., i] == 1) * power
        power *= 2
    return out


def decimal_array_to_binary_array(decimal, num_of_bits, zero_one_base=False, out=None):
    if out is None:
        out = numpy.zeros((decimal.shape[0], num_of_bits))
    for i in range(num_of_bits):
        if zero_one_base:
            out[:, i] = decimal % 2
        else:
            out[:, i] = 2 * (decimal % 2) - 1
        decimal = decimal // 2
    return out


def to_log_wave_function_vector(model, batch_size=2 ** 12, out=None):
    number_of_spins = numpy.prod(model.input_shape[1:])
    num_of_states = 2 ** number_of_spins
    if batch_size > num_of_states:
        batch_size = num_of_states
    if out is None:
        out = numpy.zeros(shape=(num_of_states,), dtype=numpy.complex128)
    for i in range(0, num_of_states, batch_size):
        batch = decimal_array_to_binary_array(numpy.arange(i, i + batch_size), number_of_spins, False).reshape(
            (batch_size,) + model.input_shape[1:])
        out[i:i + batch_size] = model.predict(batch, batch_size=batch_size)[:, 0]
    return out


def complex_norm_log_fsum_exp(arr):
    real_arr = numpy.real(arr)
    m = numpy.max(real_arr)
    return numpy.log(fsum(numpy.exp(real_arr - m))) + m


def log_fsum_exp(arr):
    m = numpy.max(arr)
    return numpy.log(fsum(numpy.exp(arr - m))) + m


def netket_vector_to_exact_variational_vector(netket_vector, netket_operator, exact_variational):
    import netket
    hilbert_index = netket.hilbert.HilbertIndex(netket_operator.hilbert)
    netket_hilbert_index = numpy.zeros(exact_variational.num_of_states, dtype=numpy.int32)
    for i in range(exact_variational.num_of_states):
        netket_hilbert_index[i] = hilbert_index.state_to_number(exact_variational.states[i, ...].reshape((-1, 1)))
    return netket_vector[netket_hilbert_index]


def vector_to_machine(wave_function_vector):
    def machine(batch):
        return wave_function_vector[binary_array_to_decimal_array(batch)][:, numpy.newaxis]

    return machine
