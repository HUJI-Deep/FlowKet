import numpy
from tensorflow.keras.callbacks import Callback

from ...optimization.exact_variational import ExactObservable


class ExactObservableCallback(Callback):
    def __init__(self, exact_variational, operator, operator_name, log_in_batch_or_epoch=True, **kwargs):
        super(ExactObservableCallback, self).__init__(**kwargs)
        self.observable = ExactObservable(exact_variational, operator)
        self.log_in_batch_or_epoch = log_in_batch_or_epoch
        self.operator_name = operator_name

    def add_observable_to_logs(self, logs):
        self.observable.update_local_energy()
        logs['Observables/%s' % self.operator_name] = numpy.real(self.observable.current_energy)

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.log_in_batch_or_epoch and ((batch % self.generator.num_of_batch_until_full_cycle) == 0):
            self.add_observable_to_logs(logs)
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if not self.log_in_batch_or_epoch:
            self.add_observable_to_logs(logs)
