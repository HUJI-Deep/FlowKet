import tensorflow
from tensorflow.keras import backend as K


def convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=True):
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    print('update_params_frequency: %s' % update_params_frequency)
    print('accumulate_sum_or_mean: %s' % accumulate_sum_or_mean)
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iterations = K.variable(0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.accumulated_iterations = accumulated_iterations

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        updates_accumulated_iterations = K.update_add(accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        if not accumulate_sum_or_mean:
            new_grads = [g / K.cast(update_params_frequency, K.dtype(g)) for g in new_grads]
        self.updated_grads = [K.update_add(p, g) for p, g in zip(self.accumulate_gradient_accumulators, new_grads)]

        def update_function():
            with tensorflow.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p))) for p in
                               self.accumulate_gradient_accumulators]
            return tensorflow.group(*(reset_grads + [updates_accumulated_iterations]))

        def just_store_function():
            return tensorflow.group(*[updates_accumulated_iterations])

        update_switch = K.equal((updates_accumulated_iterations) % update_params_frequency, 0)

        with tensorflow.control_dependencies(self.updated_grads):
            self.updates = [K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(orig_optimizer, type(orig_optimizer))
