import numpy
from tensorflow.python.keras.engine.input_layer import InputLayer


DFS_UNMAARKED = 0
TEMPORARY_MARK = 1
FULL_MARK = 2


class DependencyGraph(object):
    """docstring for Graph"""

    def __init__(self, keras_model, default_max_dependency=9):
        super(DependencyGraph, self).__init__()
        self.default_max_dependency = default_max_dependency
        self.keras_model = keras_model
        self._build_data_structures()

    def _build_data_structures(self):
        self.incoming_edges_spatial_index = {}
        self.incoming_edges_layer_index = {}
        self.nodes_dfs_status = {}
        for layer in self.keras_model.layers:
            self._add_layer_nodes(layer)

    def _add_layer_nodes(self, layer):
        output_shape = layer.output_shape
        if isinstance(layer, InputLayer):
            # we assume the last dim is channels dim in every layer
            output_shape = output_shape + (1,)
        spatial_length = numpy.prod(output_shape[1:-1])
        self.incoming_edges_spatial_index[layer] = numpy.zeros((spatial_length, self.default_max_dependency),
                                                               dtype=numpy.int32)
        self.incoming_edges_layer_index[layer] = numpy.full((spatial_length, self.default_max_dependency,), -1,
                                                            dtype=numpy.int32)
        self.nodes_dfs_status = numpy.zeros((spatial_length, ))
