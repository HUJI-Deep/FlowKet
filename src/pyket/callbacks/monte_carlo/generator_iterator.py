from tensorflow.keras.callbacks import Callback


class GeneratorIterator(Callback):
    def __init__(self, generator, **kwargs):
        super(GeneratorIterator, self).__init__(**kwargs)
        self.generator = generator

    def on_epoch_end(self, batch, logs=None):
        next(self.generator)
