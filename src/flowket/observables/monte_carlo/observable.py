import abc
import numpy


class BaseObservable(abc.ABC):
    @abc.abstractmethod
    def local_values(self, wave_function, configurations):
        pass

    def estimate(self, wave_function, configurations):
        local_values = self.local_values(wave_function, configurations)
        mean_value = numpy.mean(local_values)
        variance = numpy.var(numpy.real(local_values))
        return mean_value, variance, local_values


class LambdaObservable(BaseObservable):
    def __init__(self, observable_function):
        super(LambdaObservable, self).__init__()
        self.observable_function = observable_function

    def local_values(self, wave_function, configurations):
        return self.observable_function(wave_function, configurations)
