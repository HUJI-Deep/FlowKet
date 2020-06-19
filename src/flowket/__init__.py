import tensorflow
from .utils.v1_to_v2 import fix_tensorflow_v1_names

if not tensorflow.__version__.startswith('2'):
    fix_tensorflow_v1_names()
