import itertools


def raster(input_size):
    return itertools.product(*[range(dim_size) for dim_size in input_size])
