import math

import numpy
import tensorflow


def keras_conv_to_complex_conv(x, kernel, keras_conv, name=None):
    with tensorflow.name_scope(name, "ComplexConv", [x]) as name:
        ac = keras_conv(tensorflow.math.real(x), tensorflow.math.real(kernel))
        bd = keras_conv(tensorflow.math.imag(x), tensorflow.math.imag(kernel))
        ad = keras_conv(tensorflow.math.real(x), tensorflow.math.imag(kernel))
        bc = keras_conv(tensorflow.math.imag(x), tensorflow.math.real(kernel))
        return tensorflow.complex(ac - bd, ad + bc)


def conv2d_complex(data, filters, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1],
                   name=None):
    with tensorflow.name_scope(name, "ComplexConv2D", [data]) as name:
        def general_conv2d():
            ac = tensorflow.nn.conv2d(tensorflow.math.real(data), tensorflow.math.real(filters), strides, padding,
                                      use_cudnn_on_gpu, data_format, dilations,
                                      name='ac')
            bd = tensorflow.nn.conv2d(tensorflow.math.imag(data), tensorflow.math.imag(filters), strides, padding,
                                      use_cudnn_on_gpu, data_format, dilations,
                                      name='bd')
            ad = tensorflow.nn.conv2d(tensorflow.math.real(data), tensorflow.math.imag(filters), strides, padding,
                                      use_cudnn_on_gpu, data_format, dilations,
                                      name='ad')
            bc = tensorflow.nn.conv2d(tensorflow.math.imag(data), tensorflow.math.real(filters), strides, padding,
                                      use_cudnn_on_gpu, data_format, dilations,
                                      name='bc')
            return tensorflow.complex(ac - bd, ad + bc)

        def simplified_conv2d():
            data_shape = data.get_shape()
            reshaped_data = tensorflow.reshape(data, [-1, numpy.product(data_shape[1:])])
            filters_shape = filters.get_shape()
            reshaped_filters = tensorflow.reshape(filters, [numpy.product(filters_shape[:-1]), filters_shape[-1]])
            output_data = tensorflow.matmul(reshaped_data, reshaped_filters)
            return tensorflow.reshape(output_data, [-1, 1, 1, filters_shape[-1]])

        data_shape = data.get_shape()
        filters_shape = filters.get_shape()
        if (numpy.all(data_shape[1:] == filters_shape[:-1])
                and strides[1] == filters_shape[0] and strides[2] == filters_shape[1]):
            return simplified_conv2d()
        else:
            return general_conv2d()


def crelu(features, name=None):
    with tensorflow.name_scope(name, "CRelu", [features]) as name:
        return tensorflow.complex(tensorflow.nn.relu(tensorflow.math.real(features), name='real'),
                                  tensorflow.nn.relu(tensorflow.math.imag(features), name='imag'))


def extract_complex_image_patches(images, ksizes, strides, rates, padding, clip_imag_part=False, name=None):
    with tensorflow.name_scope(name, "ExtractImagePatches", [images]) as name:
        real_patches = tensorflow.extract_image_patches(tensorflow.math.real(images), ksizes, strides, rates, padding,
                                                        name='real')
        imag_patches = tensorflow.extract_image_patches(tensorflow.math.imag(images), ksizes, strides, rates, padding,
                                                        name='imag')
        if clip_imag_part:
            #             todo support this on gpu ?
            imag_patches = tensorflow.math.floormod(imag_patches, 2 * math.pi)
        return tensorflow.complex(real_patches, imag_patches)


def angle(number):
    real = tensorflow.math.real(number)
    imag = tensorflow.math.imag(number)
    return tensorflow.math.atan2(imag, real)


def complex_log(z):
    return tensorflow.complex(tensorflow.math.log(tensorflow.math.abs(z)), angle(z))


def lncosh(z):
    with tensorflow.name_scope("lncosh") as name:
        abs_z = tensorflow.math.abs(tensorflow.math.real(z))
        complex_abs_z = tensorflow.cast(abs_z, dtype=z.dtype)
        lncosh_res = complex_abs_z - tensorflow.cast(tensorflow.math.log(2.0), dtype=z.dtype) + complex_log(
            tensorflow.math.exp(z - complex_abs_z) + tensorflow.math.exp(-z - complex_abs_z))
        return lncosh_res


def float_norm(v):
    return tensorflow.cast(tensorflow.norm(v), tensorflow.float64)
