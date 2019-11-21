import numpy
from tensorflow.keras.callbacks import Callback


class ObservableStats(Callback):
    def __init__(self, generator, observable, observabler_name, validation_generator=None,
                 log_in_batch_or_epoch=True, validation_period=1, **kwargs):
        super(ObservableStats, self).__init__(**kwargs)
        self.observable_name = observabler_name
        self.generator = generator
        self.validation_generator = validation_generator
        self.observable = observable
        self.log_in_batch_or_epoch = log_in_batch_or_epoch
        self.validation_period = validation_period

    def add_observable_stats_to_logs(self, logs, generator, prefix=""):
        observable_value, _, _ = self.observable.estimate(generator.wave_function, generator.current_batch)
        logs['%sobservables/%s' % (prefix, self.observable_name)] = numpy.real(observable_value)

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.log_in_batch_or_epoch:
            self.add_observable_stats_to_logs(logs, self.generator)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if not self.log_in_batch_or_epoch:
            self.add_observable_stats_to_logs(logs, self.generator)
        if self.validation_generator is not None and epoch % self.validation_period == 0:
            self.add_observable_stats_to_logs(logs, self.validation_generator, prefix='val_')
