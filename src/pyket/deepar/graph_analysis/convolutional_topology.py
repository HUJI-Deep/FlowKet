import itertools

import numpy
import tensorflow
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D

from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager


class ConvolutionalTopology(LayerTopology):
    """docstring for ConvolutionalTopology"""

    def __init__(self, layer):
        super(ConvolutionalTopology, self).__init__(layer)
        if self.layer.padding != 'valid':
            raise Exception("THis topology support only valid padding, you could use "
                            "ZeroPadding1D, ZeroPadding2D, ZeroPadding3D for padding")
        self.reshaped_weights = tensorflow.reshape(self.layer.kernel, [-1, self.layer.filters])

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        flat_input = tensorflow.reshape(tensorflow.stack(dependencies_values, axis=1),
                                        shape=[-1, self.layer.input_shape[-1] * numpy.product(self.layer.kernel_size)])
        results = tensorflow.matmul(flat_input, self.reshaped_weights)
        if self.layer.use_bias:
            results = tensorflow.nn.bias_add(results, self.layer.bias)
        if self.layer.activation is not None:
            results = self.layer.activation(results)
        return results

    def get_spatial_dependency(self, spatial_location):
        dependencies = []
        for weight_location in itertools.product(*[range(dim_size) for dim_size in self.layer.kernel_size]):
            shifted_spatial_location = tuple([w * d + i * s for w, d, i, s in
                                              zip(weight_location,
                                                  self.layer.dilation_rate,
                                                  spatial_location,
                                                  self.layer.strides)])
            dependencies.append(Dependency(input_index=0, spatial_location=shifted_spatial_location))
        return dependencies


TopologyManager().register_layer_topology(Conv1D, ConvolutionalTopology)
TopologyManager().register_layer_topology(Conv2D, ConvolutionalTopology)
TopologyManager().register_layer_topology(Conv3D, ConvolutionalTopology)
