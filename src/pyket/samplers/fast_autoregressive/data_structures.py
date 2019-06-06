from collections import namedtuple

import numpy
from tensorflow.python.keras.engine.input_layer import InputLayer

GraphNode = namedtuple('GraphNode', ['layer', 'spatial_location'])
Dependency = namedtuple('Dependency', ['input_index', 'spatial_location'])

UNVISITED = 0
TEMPORARY_MARK = 1
PERMANENT_MARK = 2
NONE_INT_VALUE = -1


class DependencyGraph(object):
    """docstring for DependencyGraph"""

    def __init__(self, keras_model, default_max_number_of_outgoing_edges=10):
        super(DependencyGraph, self).__init__()
        self.default_max_number_of_outgoing_edges = default_max_number_of_outgoing_edges
        self.keras_model = keras_model
        self._create_vertices()

    def _create_vertices(self):
        self._outgoing_edges_spatial_index = {}
        self._outgoing_edges_layer_index = {}
        self._outgoing_edges_counter = {}
        self._vertices_dfs_status = {}
        self._layer_to_layer_index = {}
        self._layers_shape = {}
        for idx, layer in enumerate(self.keras_model.layers):
            self._create_layer_vertices(layer)
            self._layer_to_layer_index[layer] = idx

    def _create_layer_vertices(self, layer):
        output_shape = layer.output_shape
        if isinstance(layer, InputLayer):
            # we assume the last dim is channels dim in every layer
            output_shape = output_shape + (1,)
        self._layers_shape[layer] = output_shape[1:-1]
        spatial_size = numpy.prod(self._layers_shape[layer])
        self._outgoing_edges_spatial_index[layer] = numpy.full(
            (spatial_size, self.default_max_number_of_outgoing_edges),
            NONE_INT_VALUE, dtype=numpy.int32)
        self._outgoing_edges_layer_index[layer] = numpy.full((spatial_size, self.default_max_number_of_outgoing_edges),
                                                             NONE_INT_VALUE, dtype=numpy.int32)
        self._outgoing_edges_counter[layer] = numpy.zeros((spatial_size,),
                                                          dtype=numpy.int32)
        self._vertices_dfs_status[layer] = numpy.full((spatial_size,), UNVISITED, dtype=numpy.int32)

    def _increase_layer_num_of_edges(self, layer):
        old_incoming_edges_spatial_index_arr = self._outgoing_edges_spatial_index[layer]
        old_incoming_edges_layer_index_arr = self._outgoing_edges_layer_index[layer]
        spatial_size = old_incoming_edges_spatial_index_arr.shape[0]
        max_number_of_incoming_edges = old_incoming_edges_spatial_index_arr.shape[1]
        self._outgoing_edges_spatial_index[layer] = numpy.full((spatial_size, max_number_of_incoming_edges * 2),
                                                               NONE_INT_VALUE, dtype=numpy.int32)
        self._outgoing_edges_layer_index[layer] = numpy.full((spatial_size, max_number_of_incoming_edges * 2),
                                                             NONE_INT_VALUE, dtype=numpy.int32)
        self._outgoing_edges_spatial_index[layer][:,
        :max_number_of_incoming_edges] = old_incoming_edges_spatial_index_arr
        self._outgoing_edges_layer_index[layer][:, :max_number_of_incoming_edges] = old_incoming_edges_layer_index_arr

    def _get_layer_index(self, vertex):
        return self._layer_to_layer_index[vertex.layer]

    def _get_spatial_index(self, vertex):
        res = vertex.spatial_location[-1]
        for prev_dim_size, dim_location in zip(self._layers_shape[vertex.layer][::-1][:-1],
                                               vertex.spatial_location[::-1][1:]):
            res *= prev_dim_size
            res += dim_location
        return res

    def _spatial_index_to_location(self, spatial_index, layer_index):
        dims_location = []
        for dim_size in self._layers_shape[self.keras_model.layers[layer_index]][::-1]:
            dims_location.append(spatial_index % dim_size)
            spatial_index = spatial_index // dim_size
        return tuple(dims_location[::-1])

    def add_edge(self, from_vertex, to_vertex):
        to_vertex_spatial_index = self._get_spatial_index(to_vertex)
        to_vertex_layer_index = self._get_layer_index(to_vertex)
        from_vertex_spatial_index = self._get_spatial_index(from_vertex)
        edge_index = self._outgoing_edges_counter[from_vertex.layer][from_vertex_spatial_index]
        self._outgoing_edges_counter[from_vertex.layer][from_vertex_spatial_index] += 1
        if self._outgoing_edges_layer_index[from_vertex.layer].shape[-1] <= edge_index:
            self._increase_layer_num_of_edges(from_vertex.layer)
        self._outgoing_edges_spatial_index[from_vertex.layer][
            from_vertex_spatial_index, edge_index] = to_vertex_spatial_index
        self._outgoing_edges_layer_index[from_vertex.layer][
            from_vertex_spatial_index, edge_index] = to_vertex_layer_index

    def topological_sort(self):
        results = []
        self._reset_dfs_status()
        for vertex in self._unmarked_vertex_iterator():
            self._dfs_visitor(vertex, results)
        return [GraphNode(layer=node.layer,
                          spatial_location=self._spatial_index_to_location(node.spatial_location,
                                                                           self._layer_to_layer_index[node.layer]))
                for node in results[::-1]]

    def _reset_dfs_status(self):
        for status_arr in self._vertices_dfs_status.values():
            status_arr[:] = UNVISITED

    def _unmarked_vertex_iterator(self):
        for layer in self.keras_model.layers:
            layer_dfs_status = self._vertices_dfs_status[layer]
            for i in range(layer_dfs_status.shape[0]):
                if layer_dfs_status[i] != PERMANENT_MARK:
                    print('yield')
                    yield GraphNode(layer=layer, spatial_location=i)

    def _dfs_visitor(self, vertex, dfs_nodes_ordering):
        if self._vertices_dfs_status[vertex.layer][vertex.spatial_location] == TEMPORARY_MARK:
            raise Exception('Depedency Graph has cycle, invalid autoregressive model')
        self._vertices_dfs_status[vertex.layer][vertex.spatial_location] = TEMPORARY_MARK
        for i in range(self._outgoing_edges_counter[vertex.layer][vertex.spatial_location]):
            print(self._outgoing_edges_layer_index[vertex.layer][i])
            outgoing_layer = self.keras_model.layers[self._outgoing_edges_layer_index[vertex.layer][vertex.spatial_location, i]]
            spatial_index = self._outgoing_edges_spatial_index[vertex.layer][vertex.spatial_location, i]
            self._dfs_visitor(GraphNode(layer=outgoing_layer, spatial_location=spatial_index), dfs_nodes_ordering)
        self._vertices_dfs_status[vertex.layer][vertex.spatial_location] = PERMANENT_MARK
        dfs_nodes_ordering.append(vertex)
