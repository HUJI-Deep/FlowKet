from tensorflow.python.keras.engine.input_layer import InputLayer

from ...deepar.keras_utils.convolutional_topology import TopologyManager, ConvolutionalTopology
from ...deepar.keras_utils.sampling_topology import PlusMinusOneSamplingTopology
from ...layers import ComplexConv1D, ComplexConv2D, ComplexConv3D

TopologyManager().register_layer_topology(ComplexConv1D, ConvolutionalTopology)
TopologyManager().register_layer_topology(ComplexConv2D, ConvolutionalTopology)
TopologyManager().register_layer_topology(ComplexConv3D, ConvolutionalTopology)
TopologyManager().register_layer_topology(InputLayer, PlusMinusOneSamplingTopology)
