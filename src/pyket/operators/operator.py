import abc

import numpy

        
class Operator(abc.ABC):
    """docstring for Operator"""

    def __init__(self, hilbert_state_shape):
        super(Operator, self).__init__()
        self.hilbert_state_shape = hilbert_state_shape
        self.max_number_of_local_connections = None

    @abc.abstractmethod
    def find_conn(self, v):
        """
        Member function finding the connected elements of the Operator.  
        Starting from a given visible state v, it finds all other visible states v'  
        such that the Operator matrix element H(v,v') is different from zero.
        """
        pass

    def use_state(self, state):
        return True

    def random_states(self, num_of_states):
        """
        Member function return random state of the Operator.
        """
        return numpy.random.choice([-1, 1], size=(num_of_states, ) + self.hilbert_state_shape)


def cube_shape(number_of_spins_in_each_dimention=20, cube_dimention=1, 
               column_or_row=True):
    if cube_dimention == 1:
        return [number_of_spins_in_each_dimention, 1] if column_or_row else [1, number_of_spins_in_each_dimention]
    return [number_of_spins_in_each_dimention, ] * cube_dimention


class OperatorOnGrid(Operator, abc.ABC):
    """docstring for OperatorOnGrid"""
    def __init__(self, hilbert_state_shape=cube_shape(), pbc=True):
        super(OperatorOnGrid, self).__init__(hilbert_state_shape)
        self.pbc = pbc