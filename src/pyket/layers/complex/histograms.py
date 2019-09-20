import math

import tensorflow
from tensorflow.keras.layers import Layer


class LogSpaceComplexNumberHistograms(Layer):
    def __init__(self, **kwargs):
        super(LogSpaceComplexNumberHistograms, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        magnitude_hist = tensorflow.summary.histogram("magnitude", 
            2.0 * tensorflow.math.real(x))
        phase_hist = tensorflow.summary.histogram("phase", 
            tensorflow.math.imag(x))
        normalized_phase_hist = tensorflow.summary.histogram("normalized_phase", 
            tensorflow.math.imag(x) %  (2.0 * math.pi))
        return x
