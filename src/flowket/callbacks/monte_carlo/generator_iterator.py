from tensorflow.keras.callbacks import Callback


class GeneratorIterator(Callback):
    def __init__(self, generator, period=1, **kwargs):
        super(GeneratorIterator, self).__init__(**kwargs)
        self.generator = generator
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            next(self.generator)
