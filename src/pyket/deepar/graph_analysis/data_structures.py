from collections import namedtuple

GraphNode = namedtuple('GraphNode', ['layer', 'spatial_location', 'output_index'])
Dependency = namedtuple('Dependency', ['input_index', 'spatial_location'])