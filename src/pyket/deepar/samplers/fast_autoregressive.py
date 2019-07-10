import networkx
import numpy
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.input_layer import InputLayer

from .base_sampler import Sampler
from ..graph_analysis.dependency_graph import DependencyGraph
from ..graph_analysis.topology_manager import TopologyManager


class FastAutoregressiveSampler(Sampler):
    """docstring for FastAutoregressiveSampler"""

    def __init__(self, conditional_log_probs_machine, batch_size, **kwargs):
        super(FastAutoregressiveSampler, self).__init__(input_size=conditional_log_probs_machine.input_shape[1:],
                                                        batch_size=batch_size, **kwargs)
        self.input_layer = conditional_log_probs_machine.get_layer(conditional_log_probs_machine.input_names[0])
        self.batch_size_t = K.placeholder(dtype='int32', shape=())
        self.dependencies_graph = DependencyGraph(conditional_log_probs_machine)
        self._layer_to_activation_array = {}
        self._build_sampling_function()

    def __next__(self):
        if self.mini_batch_size < self.batch_size:
            return numpy.concatenate([self.sampling_function(self.mini_batch_size)[0]
                                      for _ in range(self.batch_size // self.mini_batch_size)])

        return self.sampling_function(self.mini_batch_size)[0]

    def _create_layer_activation_array(self, layer):
        output_shape = layer.output_shape
        if isinstance(layer, InputLayer):
            # we assume the last dim is channels dim in every layer
            output_shape = output_shape + (1,)
        zeros = tensorflow.zeros(shape=(self.batch_size_t, output_shape[-1],),
                                 dtype=tensorflow.as_dtype(layer.dtype))
        self._layer_to_activation_array[layer] = numpy.full(output_shape[1:-1], fill_value=zeros)

    def _get_layer_activation_array(self, layer):
        return self._layer_to_activation_array[layer]

    def _get_or_create_layer_activation_array(self, layer):
        if layer not in self._layer_to_activation_array:
            self._create_layer_activation_array(layer)
        return self._get_layer_activation_array(layer)

    def _get_dependency_value(self, layer, dependency):
        layer_inputs = self.dependencies_graph.layer_to_input_layers[layer]
        return self._get_or_create_layer_activation_array(layer_inputs[dependency.input_index])[
            dependency.spatial_location]

    def _extract_the_sample(self):
        sample_tensors_array = self._get_layer_activation_array(self.input_layer)
        flat_sample = tensorflow.stack(sample_tensors_array.flatten().tolist(), axis=1)
        return tensorflow.reshape(flat_sample, (-1,) + sample_tensors_array.shape)

    def _build_sampling_function(self):
        self.sampling_order = list(networkx.topological_sort(self.dependencies_graph.graph))
        for node in self.sampling_order:
            layer_topology = TopologyManager().get_layer_topology(node.layer)
            dependencies = layer_topology.get_spatial_dependency(node.spatial_location)
            activation_array = self._get_or_create_layer_activation_array(node.layer)
            if len(dependencies) == 0:
                dependencies_values = self.batch_size_t
            else:
                dependencies_values = [self._get_dependency_value(node.layer, dependency) for dependency in dependencies]
            activation_array[node.spatial_location] = layer_topology. \
                apply_layer_for_single_spatial_location(node.spatial_location, dependencies_values)
        sample = self._extract_the_sample()
        self.sampling_function = K.function(inputs=[self.batch_size_t], outputs=[sample])
