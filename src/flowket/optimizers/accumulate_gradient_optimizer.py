import tensorflow
from tensorflow.keras import backend as K


def convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=True, ema_decay=0):
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    print('update_params_frequency: %s' % update_params_frequency)
    if accumulate_sum_or_mean:
        print('using sum of gradients')
    else:
        print('using gradients mean')
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    orig_optimizer.accumulated_iterations = K.variable(0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.update_params_frequency = K.variable(update_params_frequency, dtype='int64', name='update_params_frequency')
    if ema_decay > 0:
        orig_optimizer.total_iterations = K.variable(0, dtype=K.floatx(), name='total_iterations')

    def set_update_params_frequency(self, update_params_frequency):
        K.set_value(self.update_params_frequency, update_params_frequency)

    def set_weights_ema(self):
        return tensorflow.group(*[K.update(p, e_p / (1 - K.pow(ema_decay, self.total_iterations))) for e_p, p in zip(self.params_ema, self.params_for_ema_tracking)])

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name=p.name + '_gradient_accumulators') for p in params]
        if ema_decay > 0:
            self.params_for_ema_tracking = params
            self.params_ema = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name=p.name + '_ema') for p in params]
        updates_accumulated_iterations = K.update_add(self.accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        if not accumulate_sum_or_mean:
            new_grads = [g / K.cast(self.update_params_frequency, K.dtype(g)) for g in new_grads]
        self.updated_grads = [K.update_add(p, g) for p, g in zip(self.accumulate_gradient_accumulators, new_grads)]


        def update_function():
            updated_params = pass
            with tensorflow.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p))) for p in
                                self.accumulate_gradient_accumulators]
                if ema_decay > 0:
                    reset_grads += [K.update_add(self.total_iterations, 1)]
                    reset_grads += [K.update(e_p, (e_p * ema_decay) + (1 - ema_decay) * p) for e_p, p in zip(self.params_ema, updated_params)]
            return tensorflow.group(*(reset_grads + [updates_accumulated_iterations]))

        def just_store_function():
            return tensorflow.group(*[updates_accumulated_iterations])

        update_switch = K.equal((updates_accumulated_iterations) % self.update_params_frequency, 0)

        with tensorflow.control_dependencies(self.updated_grads):
            self.updates = [K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.set_update_params_frequency = set_update_params_frequency.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.set_weights_ema = set_weights_ema.__get__(orig_optimizer, type(orig_optimizer))