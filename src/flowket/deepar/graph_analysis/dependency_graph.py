import itertools

import networkx
from tensorflow.python.keras.engine.input_layer import InputLayer

from .data_structures import GraphNode
from .topology_manager import TopologyManager


def visit_layer_predecessors(layer, visitor, visited_layers=None, layer_output_index=0):
    if visited_layers is None:
        visited_layers = set()
    visited_layers.add((layer_output_index, layer))
    layer_nodes = layer.inbound_nodes
    assert len(layer_nodes) == 1
    inbound_layers = layer_nodes[-1].inbound_layers
    if not isinstance(inbound_layers, list):
        inbound_layers = [inbound_layers]
    tensor_indices = layer_nodes[-1].tensor_indices
    if not isinstance(tensor_indices, list):
        tensor_indices = [tensor_indices]
    visitor(layer, layer_output_index, inbound_layers, tensor_indices)
    for tensor_indice, inbound_layer in zip(tensor_indices, inbound_layers):
        if (tensor_indice, inbound_layer) not in visited_layers:
            visit_layer_predecessors(inbound_layer, visitor, visited_layers, layer_output_index=tensor_indice)


def assert_valid_probabilistic_model(model):
    try:
        networkx.topological_sort(DependencyGraph(model).graph)
    except networkx.NetworkXUnfeasible:
        raise Exception("This model don't have valid autoregressive ordering")


class DependencyGraph(object):
    """docstring for DependencyGraph"""

    def __init__(self, autoregressive_model):
        super(DependencyGraph, self).__init__()
        self.model = autoregressive_model
        self.layer_to_output_shape = {}
        self.layer_to_input_layers = {}
        self.layer_to_input_indices = {}
        self.input_layer = self.model.get_layer(self.model.input_names[0])
        self.output_layer = self.model.get_layer(self.model.output_names[0])
        self._calculate_layers_output_shape()
        self._build_graph()

    def _calculate_layers_output_shape(self):
        for layer in self.model.layers:
            output_shape = layer.get_output_shape_at(0)
            if isinstance(layer, InputLayer):
                # we assume the last dim is channels dim in every layer
                output_shape = output_shape + (1,)
            if isinstance(output_shape, tuple):
                output_shape = [output_shape]
            self.layer_to_output_shape[layer] = [o[1:-1]for o in output_shape]

    def _dependency_graph_visitor(self, layer, layer_output_index, inbound_layers, inbound_indices):
        self.layer_to_input_layers[layer] = inbound_layers
        self.layer_to_input_indices[layer] = inbound_indices
        if len(inbound_layers) == 0:
            return
        self._add_inbound_layers_dependencies(layer, layer_output_index, inbound_layers, inbound_indices)

    def _add_inbound_layers_dependencies(self, layer, layer_output_index, inbound_layers, inbound_indices):
        output_shape = self.layer_to_output_shape[layer][layer_output_index]
        for spatial_location in itertools.product(*[range(dim_shape) for dim_shape in
                                                    output_shape]):
            self._add_spatial_location_dependencies(layer, layer_output_index, inbound_layers, inbound_indices, spatial_location)

    def _add_spatial_location_dependencies(self, layer, layer_output_index, inbound_layers, inbound_indices, spatial_location):
        spatial_dependencies = TopologyManager().get_layer_topology(layer).\
            get_spatial_dependency(spatial_location, output_index=layer_output_index)
        current_node = GraphNode(layer=layer, spatial_location=spatial_location, output_index=layer_output_index)
        for dependency in spatial_dependencies:
            incoming_node = GraphNode(layer=inbound_layers[dependency.input_index],
                                      spatial_location=dependency.spatial_location,
                                      output_index=inbound_indices[dependency.input_index])
            self.graph.add_edge(incoming_node, current_node)

    def _add_sampling_probabilities_dependencies(self):
        self.layer_to_input_layers[self.input_layer] = [self.output_layer]
        self.layer_to_input_indices[self.input_layer] = [0]
        for spatial_location in itertools.product(*[range(dim_shape) for dim_shape in
                                                    self.model.output_shape[1:-1]]):
            input_node = GraphNode(layer=self.input_layer, spatial_location=spatial_location, output_index=0)
            probabilities_node = GraphNode(layer=self.output_layer, spatial_location=spatial_location, output_index=0)
            self.graph.add_edge(probabilities_node, input_node)

    def _build_graph(self):
        self.graph = networkx.DiGraph()
        visit_layer_predecessors(self.output_layer, visitor=self._dependency_graph_visitor)
        self._add_sampling_probabilities_dependencies()
