from .netket_operator import NetketOperatorWrapper

import numpy
import netket


def j1j2_two_dim_netket_operator(number_of_spins, j2=0.5, pbc=False):
    if isinstance(number_of_spins, tuple):
        L1, L2 = number_of_spins
    else:
        L1, L2 = number_of_spins, number_of_spins
    J = [1.0, j2]
    edge_colors = []
    for h in range(L1):
        for w in range(L2 - 1):
            edge_colors.append([w + L2 * h, w + 1 + L2 * h, 1])
            if h < L1 - 1:
                edge_colors.append([w + L2 * h, w + L2 * (h + 1), 1])
                edge_colors.append([w + L2 * h, w + 1 + L2 * (h + 1), 2])
            elif pbc:
                edge_colors.append([w + L2 * h, w, 1])
                edge_colors.append([w + L2 * h, w + 1, 2])
            if h > 0:
                edge_colors.append([w + L2 * h, w + 1 + L2 * (h - 1), 2])
            elif pbc:
                edge_colors.append([w + L2 * h, w + 1 + L2 * (L1 - 1), 2])
        w = L2 - 1
        if pbc:
            edge_colors.append([L2 - 1 + L2 * h, L2 * h, 1])
            edge_colors.append([w + L2 * h, L2 * ((h + 1) % L1), 2])
            edge_colors.append([w + L2 * h, L2 * ((L1 + h - 1) % L1), 2])
        if h < L1 - 1:
            edge_colors.append([w + L2 * h, w + L2 * (h + 1), 1])
        elif pbc:
            edge_colors.append([w + L2 * h, w, 1])

    g = netket.graph.CustomGraph(edge_colors)
    if L1 * L2 % 2 == 0:
        hi = netket.hilbert.Spin(s=0.5, total_sz=0.0, graph=g)
    else:
        hi = netket.hilbert.Spin(s=0.5,  graph=g)
    sigmaz = [[1, 0], [0, -1]]
    sigmax = [[0, 1], [1, 0]]
    sigmay = [[0, -1j], [1j, 0]]

    interaction = numpy.kron(sigmaz, sigmaz) + numpy.kron(sigmax, sigmax) + numpy.kron(sigmay, sigmay)

    bond_operator = [
        (J[0] * interaction).tolist(),
        (J[1] * interaction).tolist(),
    ]

    bond_color = [1, 2]

    return netket.operator.GraphOperator(hi, bondops=bond_operator, bondops_colors=bond_color)


def j1j2_two_dim_operator(hilbert_state_shape, j2=0.5, pbc=True):
    max_number_of_local_connections = numpy.prod(hilbert_state_shape) * len(hilbert_state_shape) * 2 + 1
    assert len(hilbert_state_shape) == 2
    return NetketOperatorWrapper(
        j1j2_two_dim_netket_operator(tuple(hilbert_state_shape),
                                     j2=j2,
                                     pbc=pbc),
        hilbert_state_shape=hilbert_state_shape,
        should_calc_unused=True, max_number_of_local_connections=max_number_of_local_connections)


J1J2 = j1j2_two_dim_operator
