from tensorflow.keras.callbacks import Callback


class MachineUpdated(Callback):
    def __init__(self, generator, **kwargs):
        super(MachineUpdated, self).__init__(**kwargs)
        self.generator = generator

    def on_batch_end(self, batch, logs=None):
        self.generator.machine_updated()
