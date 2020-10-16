import tensorflow
if tensorflow.__version__.startswith('2'):
    from .accumulate_gradient_optimizer_v2 import convert_to_accumulate_gradient_optimizer
else:
    from .accumulate_gradient_optimizer_v1 import convert_to_accumulate_gradient_optimizer
