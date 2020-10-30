import pickle
import os
import time

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


def save_optimizer_weights(model, filepath, epoch, batch):
    symbolic_weights = getattr(model.optimizer, 'weights')
    if symbolic_weights is None:
        return
    weight_values = K.batch_get_value(symbolic_weights)
    with open(filepath + '_optimizer.pkl', 'wb') as f:
        pickle.dump((weight_values, epoch, batch), f)


def load_optimizer_weights(model, filepath):
    filepath += '_optimizer.pkl'
    if not os.path.exists(filepath):
        return
    with open(filepath, 'rb') as f:
        weight_values, epoch, batch = pickle.load(f)
    model._make_train_function()
    model.optimizer.set_weights(weight_values)
    return epoch, batch


class CheckpointByTime(Callback):
    """Save the model every x minutes.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
    """

    def __init__(self, filepath, save_frequency_in_minutes=30, save_weights_only=False):
        super(CheckpointByTime, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.save_frequency_in_minutes = save_frequency_in_minutes
        self.last_save_time = time.time()
        self.current_epoch = 0

    def _save(self, logs, batch):
        filepath = self.filepath.format(**logs)
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
            save_optimizer_weights(self.model, filepath, self.current_epoch, batch)
        else:
            self.model.save(filepath, overwrite=True)
        self.last_save_time = time.time()
        
    def on_batch_end(self, batch, logs=None):
        batch_end_time = time.time()
        if batch_end_time - self.last_save_time < self.save_frequency_in_minutes * 60:
            return
        logs = logs or {}
        self._save(logs, batch)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_end(self, logs=None):
        logs = logs or {}
        self.current_epoch += 1
        self._save(logs, 0)
