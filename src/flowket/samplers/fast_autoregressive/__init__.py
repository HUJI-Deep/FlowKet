from tensorflow.python.keras.engine.input_layer import InputLayer

from ...deepar.samplers import FastAutoregressiveSampler
from ...deepar.graph_analysis.convolutional_topology import TopologyManager
from ...deepar.graph_analysis.sampling_topology import PlusMinusOneSamplingTopology


TopologyManager().register_layer_topology(InputLayer, PlusMinusOneSamplingTopology)
