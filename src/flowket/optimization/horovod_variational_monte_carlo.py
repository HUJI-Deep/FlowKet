import tensorflow
import numpy
import horovod.tensorflow as hvd

from tensorflow.python.keras import backend as K

from .variational_monte_carlo import VariationalMonteCarlo


class HorovodVariationalMonteCarlo(VariationalMonteCarlo):
    def __init__(self, model, operator, sampler, **kwargs):
        super(HorovodVariationalMonteCarlo, self).__init__(model, operator, sampler, **kwargs)
        with K.name_scope('HorovodVariationalMonteCarlo'):
            self.current_energy_real_t = K.variable(0, name="current_energy_real", dtype=tensorflow.float64)
            self.current_energy_imag_t = K.variable(0, name="current_energy_imag", dtype=tensorflow.float64)
            K.get_session().run([self.current_energy_real_t.initializer, self.current_energy_imag_t.initializer,])
            self.current_energy_real_allreduce_op = hvd.allreduce(self.current_energy_real_t)
            self.current_energy_imag_allreduce_op = hvd.allreduce(self.current_energy_imag_t)
    
    def _update_batch_local_energy(self):
        self.current_energy, self.current_local_energy_variance, self.current_local_energy = \
            self.energy_observable.estimate(self.wave_function, self.current_batch)
        K.set_value(self.current_energy_real_t, numpy.real(self.current_energy))
        K.set_value(self.current_energy_imag_t, numpy.imag(self.current_energy))
        reduced_energy  = K.get_session().run([self.current_energy_real_allreduce_op, self.current_energy_imag_allreduce_op])
        self.current_energy = reduced_energy[0] + 1.j* reduced_energy[1]
