from tensorflow.keras.layers import Lambda


class LambdaWithOneToOneTopology(Lambda):
    def __init__(self, function, **kwargs):
        super(LambdaWithOneToOneTopology, self).__init__(function, **kwargs)
