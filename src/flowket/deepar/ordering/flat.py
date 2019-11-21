def to_flat_ordering(ordering, shape):
    for i in ordering:
        current_idx = 0
        for dim_idx, dim_size in zip(i, shape):
            current_idx *= dim_size
            current_idx += dim_idx
        yield current_idx


def to_flat_inverse_ordering(ordering, shape):
    inversed_order = [0] * len(ordering)
    for i, p in enumerate(to_flat_ordering(ordering, shape)):
        inversed_order[p] = i
    return inversed_order
