import itertools

import numpy
import networkx
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.input_layer import InputLayer

from .data_structures import GraphNode
from .topology_manager import TopologyManager
from ..base_sampler import Sampler


def visit_layer_predecessors(layer, visitor, visited_layers=None):
    if visited_layers is None:
        visited_layers = set()
    visited_layers.add(layer)
    layer_nodes = layer.inbound_nodes
    assert len(layer_nodes) == 1
    inbound_layers = layer_nodes[-1].inbound_layers
    visitor(layer, inbound_layers)
    for inbound_layer in inbound_layers:
        if inbound_layer not in visited_layers:
            visit_layer_predecessors(inbound_layer, visitor, visited_layers)


class FastAutoregressiveSampler(Sampler):
    """docstring for FastAutoregressiveSampler"""

    def __init__(self, conditional_log_probs_machine, batch_size, **kwargs):
        super(FastAutoregressiveSampler, self).__init__(input_size=conditional_log_probs_machine.input_shape[1:],
                                                        batch_size=batch_size, **kwargs)
        self.model = conditional_log_probs_machine
        self.input_layer = self.model.get_layer(self.model.input_names[0])
        self.output_layer = self.model.get_layer(self.model.output_names[0])
        self._layer_to_activation_array = {}
        self._layer_to_input_layers = {}
        self._build_dependency_graph()
        self._build_sampling_function()
        self._fake_input = numpy.zeros((self.batch_size, )).tolist()

    def __next__(self):
        return self.sampling_function(self._fake_input)[0]

    def _create_layer_activation_array(self, layer):
        output_shape = layer.output_shape
        if isinstance(layer, InputLayer):
            # we assume the last dim is channels dim in every layer
            output_shape = output_shape + (1,)
        zeros = tensorflow.zeros(shape=(self.batch_size, output_shape[-1],), dtype=tensorflow.as_dtype(layer.dtype))
        self._layer_to_activation_array[layer] = numpy.full(output_shape[1:-1], fill_value=zeros)

    def _get_layer_activation_array(self, layer):
        return self._layer_to_activation_array[layer]

    def _get_or_create_layer_activation_array(self, layer):
        if layer not in self._layer_to_activation_array:
            self._create_layer_activation_array(layer)
        return self._get_layer_activation_array(layer)

    def _dependency_graph_visitor(self, layer, inbound_layers):
        self._get_or_create_layer_activation_array(layer)
        self._layer_to_input_layers[layer] = inbound_layers
        if len(inbound_layers) == 0:
            return
        self._add_inbound_layers_dependencies(layer, inbound_layers)

    def _add_inbound_layers_dependencies(self, layer, inbound_layers):
        activation_array = self._get_or_create_layer_activation_array(layer)
        for spatial_location in itertools.product(*[range(dim_shape) for dim_shape in
                                                    activation_array.shape]):
            self._add_spatial_location_dependencies(layer, inbound_layers, spatial_location)

    def _add_spatial_location_dependencies(self, layer, inbound_layers, spatial_location):
        spatial_dependencies = TopologyManager().get_layer_topology(layer).get_spatial_dependency(spatial_location)
        current_node = GraphNode(layer=layer, spatial_location=spatial_location)
        for dependency in spatial_dependencies:
            incoming_node = GraphNode(layer=inbound_layers[dependency.input_index],
                                      spatial_location=dependency.spatial_location)
            self.dependencies_graph.add_edge(incoming_node, current_node)

    def _add_sampling_probabilities_dependencies(self):
        self._layer_to_input_layers[self.input_layer] = [self.output_layer]
        for spatial_location in itertools.product(*[range(dim_shape) for dim_shape in
                                                    self.model.output_shape[1:-1]]):
            input_node = GraphNode(layer=self.input_layer, spatial_location=spatial_location)
            probabilities_node = GraphNode(layer=self.output_layer, spatial_location=spatial_location)
            self.dependencies_graph.add_edge(probabilities_node, input_node)

    def _build_dependency_graph(self):
        self.dependencies_graph = networkx.DiGraph()
        visit_layer_predecessors(self.output_layer, visitor=self._dependency_graph_visitor)
        self._add_sampling_probabilities_dependencies()

    def _get_dependency_value(self, layer, dependency):
        layer_inputs = self._layer_to_input_layers[layer]
        return self._get_or_create_layer_activation_array(layer_inputs[dependency.input_index])[
            dependency.spatial_location]

    def _extract_the_sample(self):
        sample_tensors_array = self._get_layer_activation_array(self.input_layer)
        flat_sample = tensorflow.stack(sample_tensors_array.flatten().tolist(), axis=1)
        return tensorflow.reshape(flat_sample, (-1,) + sample_tensors_array.shape)

    def _build_sampling_function(self):
        self.sampling_order = list(networkx.topological_sort(self.dependencies_graph))
        for node in self.sampling_order:
            layer_topology = TopologyManager().get_layer_topology(node.layer)
            dependencies = layer_topology.get_spatial_dependency(node.spatial_location)
            if len(dependencies) == 0:
                continue
            activation_array = self._get_or_create_layer_activation_array(node.layer)
            dependencies_values = [self._get_dependency_value(node.layer, dependency) for dependency in dependencies]
            activation_array[node.spatial_location] = layer_topology. \
                apply_layer_for_single_spatial_location(node.spatial_location, dependencies_values)
        sample = self._extract_the_sample()
        self.sampling_function = K.function(inputs=[], outputs=[sample])
