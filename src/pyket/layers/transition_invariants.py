import tensorflow

from tensorflow.keras.layers import Layer


class Roll(Layer):
    def __init__(self, roll_for_axis, axes_to_roll=None, **kwargs):
        super(Roll, self).__init__(**kwargs)
        if axes_to_roll is None:
            axes_to_roll = list(range(1, 1 + len(roll_for_axis)))
        assert len(axes_to_roll) == len(roll_for_axis)
        self.roll_for_axis = roll_for_axis
        self.axes_to_roll = axes_to_roll

    def call(self, x, mask=None):
        return tensorflow.roll(x, self.roll_for_axis, self.axes_to_roll)
        
    def get_config(self):
        config = {'roll_for_axis': self.roll_for_axis,
                  'axes_to_roll': self.axes_to_roll}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))