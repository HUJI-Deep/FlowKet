from tensorflow.keras.layers import Concatenate


class ConcatenateChannels(Concatenate):
    def __init__(self, **kwargs):
        super(ConcatenateChannels, self).__init__(axis=-1, **kwargs)