import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class PeriodicPadding(Layer):
    def __init__(self, padding_size, horizontal_padding=True, vertical_padding=True, **kwargs):
        super(PeriodicPadding, self).__init__(**kwargs)
        self.padding_size = padding_size
        self.horizontal_padding = horizontal_padding
        self.vertical_padding = vertical_padding

    def call(self, x, mask=None):
        '''
        Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
        '''
        # copy from https://github.com/tensorflow/tensorflow/issues/956 
        # todo check the channels axis
        upper_pad = x[:,-self.padding_size:,:,:]
        lower_pad = x[:,:self.padding_size,:,:]
        
        if self.vertical_padding:
            x = K.concatenate([upper_pad, x, lower_pad], axis=1)
        
        left_pad = x[:,:,-self.padding_size:,:]
        right_pad = x[:,:,:self.padding_size,:]
        
        if self.horizontal_padding:
            x = K.concatenate([left_pad, x, right_pad], axis=2)
        
        return x

    def get_config(self):
        config = {
            'padding_size': self.padding_size,
            'horizontal_padding': self.horizontal_padding,
            'vertical_padding': self.vertical_padding
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

