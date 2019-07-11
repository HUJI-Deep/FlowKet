import itertools

import networkx
from tensorflow.python.keras.engine.input_layer import InputLayer

from .data_structures import GraphNode
from .topology_manager import TopologyManager


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
        self.input_layer = self.model.get_layer(self.model.input_names[0])
        self.output_layer = self.model.get_layer(self.model.output_names[0])
        self._calculate_layers_output_shape()
        self._build_graph()

    def _calculate_layers_output_shape(self):
        for layer in self.model.layers:
            output_shape = layer.output_shape
            if isinstance(layer, InputLayer):
                # we assume the last dim is channels dim in every layer
                output_shape = output_shape + (1,)
            if isinstance(output_shape, list):
                # todo handel multiple outputs we can find the real dependency with Node.tensor_indices
                pass
            else:
                self.layer_to_output_shape[layer] = output_shape[1:-1]

    def _dependency_graph_visitor(self, layer, inbound_layers):
        self.layer_to_input_layers[layer] = inbound_layers
        if len(inbound_layers) == 0:
            return
        self._add_inbound_layers_dependencies(layer, inbound_layers)

    def _add_inbound_layers_dependencies(self, layer, inbound_layers):
        output_shape = self.layer_to_output_shape[layer]
        for spatial_location in itertools.product(*[range(dim_shape) for dim_shape in
                                                    output_shape]):
            self._add_spatial_location_dependencies(layer, inbound_layers, spatial_location)

    def _add_spatial_location_dependencies(self, layer, inbound_layers, spatial_location):
        spatial_dependencies = TopologyManager().get_layer_topology(layer).get_spatial_dependency(spatial_location)
        current_node = GraphNode(layer=layer, spatial_location=spatial_location)
        for dependency in spatial_dependencies:
            incoming_node = GraphNode(layer=inbound_layers[dependency.input_index],
                                      spatial_location=dependency.spatial_location)
            self.graph.add_edge(incoming_node, current_node)

    def _add_sampling_probabilities_dependencies(self):
        self.layer_to_input_layers[self.input_layer] = [self.output_layer]
        for spatial_location in itertools.product(*[range(dim_shape) for dim_shape in
                                                    self.model.output_shape[1:-1]]):
            input_node = GraphNode(layer=self.input_layer, spatial_location=spatial_location)
            probabilities_node = GraphNode(layer=self.output_layer, spatial_location=spatial_location)
            self.graph.add_edge(probabilities_node, input_node)

    def _build_graph(self):
        self.graph = networkx.DiGraph()
        visit_layer_predecessors(self.output_layer, visitor=self._dependency_graph_visitor)
        self._add_sampling_probabilities_dependencies()
