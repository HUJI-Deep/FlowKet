import copy
import itertools

import networkx
import numpy
import tensorflow
from tensorflow.keras import backend as K

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

    def copy_with_new_batch_size(self, batch_size, mini_batch_size=None):
        new_sampler = copy.copy(self)
        new_sampler._set_batch_size(batch_size, mini_batch_size)
        return new_sampler

    def __next__(self):
        if self.mini_batch_size < self.batch_size:
            return numpy.concatenate([self.sampling_function([self.mini_batch_size])[0]
                                      for _ in range(self.batch_size // self.mini_batch_size)])

        return self.sampling_function([self.mini_batch_size])[0]

    def _create_layer_activation_array(self, layer):
        self._layer_to_activation_array[layer] = []
        for output_index, output_shape in enumerate(self.dependencies_graph.layer_to_output_shape[layer]):
            zeros = TopologyManager().get_layer_topology(layer).get_zeros(self.batch_size_t, output_index)
            activation_array = numpy.empty(output_shape, dtype=object)
            for i in itertools.product(*[range(s) for s in output_shape]):
                activation_array[i] = zeros
            self._layer_to_activation_array[layer].append(activation_array)

    def _get_layer_activation_array(self, layer, output_index):
        return self._layer_to_activation_array[layer][output_index]

    def _get_or_create_layer_activation_array(self, layer, output_index):
        if layer not in self._layer_to_activation_array:
            self._create_layer_activation_array(layer)
        return self._get_layer_activation_array(layer, output_index)

    def _get_dependency_value(self, layer, dependency):
        layer_inputs = self.dependencies_graph.layer_to_input_layers[layer]
        inputs_indices = self.dependencies_graph.layer_to_input_indices[layer]
        return self._get_or_create_layer_activation_array(layer_inputs[dependency.input_index],
                                                          inputs_indices[dependency.input_index])[
            dependency.spatial_location]

    def _extract_the_sample(self):
        sample_tensors_array = self._get_layer_activation_array(self.input_layer, output_index=0)
        flat_sample = tensorflow.stack(sample_tensors_array.flatten().tolist(), axis=1)
        return tensorflow.reshape(flat_sample, (-1,) + sample_tensors_array.shape)

    def _build_sampling_function(self):
        self.sampling_order = list(networkx.topological_sort(self.dependencies_graph.graph))
        for node in self.sampling_order:
            layer_topology = TopologyManager().get_layer_topology(node.layer)
            dependencies = layer_topology.get_spatial_dependency(node.spatial_location, output_index=node.output_index)
            activation_array = self._get_or_create_layer_activation_array(node.layer, node.output_index)
            if len(dependencies) == 0:
                dependencies_values = self.batch_size_t
            else:
                dependencies_values = [self._get_dependency_value(node.layer, dependency) for dependency in dependencies]
            activation_array[node.spatial_location] = layer_topology. \
                apply_layer_for_single_spatial_location(node.spatial_location, dependencies_values)
        sample = self._extract_the_sample()
        self.sampling_function = K.function(inputs=[self.batch_size_t], outputs=[sample])
