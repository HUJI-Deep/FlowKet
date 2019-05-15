from collections import namedtuple


GraphNode = namedtuple('GraphNode', ['layer', 'spatial_location'])
Dependency = namedtuple('Dependency', ['input_index', 'spatial_location'])
