def to_flat_ordering(ordering, shape):
    for i in ordering:
        current_idx = 0
        for dim_idx, dim_size in zip(i, shape):
            current_idx *= dim_size
            current_idx += dim_idx
        yield current_idx
