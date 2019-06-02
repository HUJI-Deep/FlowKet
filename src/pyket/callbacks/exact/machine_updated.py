from tensorflow.keras.callbacks import Callback


class MachineUpdated(Callback):
    def __init__(self, exact_variational, **kwargs):
        super(MachineUpdated, self).__init__(**kwargs)
        self.exact_variational = exact_variational

    def on_batch_end(self, batch, logs=None):
        self.exact_variational.machine_updated()
