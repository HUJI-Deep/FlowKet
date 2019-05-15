import itertools

from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager
from ...layers import ComplexConv2D

import numpy
import tensorflow
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D


class ConvolutionalTopology(LayerTopology):
    """docstring for Conv1DTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        flat_input = tensorflow.reshape(tensorflow.stack(dependencies_values, axis=1),
                                        shape=[-1, self.layer.input_shape[-1] * numpy.product(self.layer.kernel_size)])
        reshaped_weights = tensorflow.reshape(self.layer.kernel, [-1, self.layer.filters])
        results = tensorflow.matmul(flat_input, reshaped_weights)
        if self.layer.use_bias:
            results = tensorflow.nn.bias_add(results, self.layer.bias)
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


TopologyManager().register_layer_topology(ConvolutionalTopology, Conv1D)
TopologyManager().register_layer_topology(ConvolutionalTopology, Conv2D)
TopologyManager().register_layer_topology(ConvolutionalTopology, Conv3D)
TopologyManager().register_layer_topology(ConvolutionalTopology, ComplexConv2D)
