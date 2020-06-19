import tensorflow
if tensorflow.__version__.startswith('2'):
    from tensorflow.keras.callbacks import TensorBoard
else:
    from ..tensorboard import TensorBoard
import numpy


class TensorBoardWithGeneratorValidationData(TensorBoard):
    """docstring for TensorBoardWithGeneratorValidationData"""
    def __init__(self, generator, **kwargs):
        super(TensorBoardWithGeneratorValidationData, self).__init__(**kwargs)
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        validation_size = self.generator.current_batch.shape[0]
        self.validation_data = self.generator.current_batch, \
                               numpy.zeros((validation_size, 1), dtype=numpy.complex128), \
                               numpy.full(validation_size, fill_value=1.0/validation_size)
        super().on_epoch_end(epoch, logs)
