import numpy
from tensorflow.keras.callbacks import Callback

from ...optimization.variational_monte_carlo import Observable


class ObservableStats(Callback):
    def __init__(self, generator, operator, operator_name, validation_generator=None,
                 log_in_batch_or_epoch=True, **kwargs):
        super(ObservableStats, self).__init__(**kwargs)
        self.operator_name = operator_name
        self.observable = Observable(generator, operator)
        self.validation_observable = Observable(validation_generator, operator)
        self.log_in_batch_or_epoch = log_in_batch_or_epoch

    def add_observable_stats_to_logs(self, logs, observable, prefix=""):
        logs['%sobservables/%s' % (prefix, self.operator_name)] = numpy.real(observable.current_energy)

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.log_in_batch_or_epoch:
            self.add_observable_stats_to_logs(logs, self.observable)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if not self.log_in_batch_or_epoch:
            self.add_observable_stats_to_logs(logs, self.observable)
        if self.validation_observable is not None:
            self.add_observable_stats_to_logs(logs, self.validation_observable, prefix='val_')
