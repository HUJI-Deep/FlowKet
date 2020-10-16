import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class AccumulateGradientOptimizerWrapper(optimizer_v2.OptimizerV2):
    def __init__(self, optimizer, update_params_frequency, accumulate_sum_or_mean=True, ema_decay=0, name="AccumulateGradient", **kwargs):
        super(AccumulateGradientOptimizerWrapper, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        if not isinstance(update_params_frequency, int):
            raise TypeError("update_params_frequency must be of int type")

        if not isinstance(accumulate_sum_or_mean, bool):
            raise TypeError("accumulate_sum_or_mean must be of bool type")

        self._optimizer = optimizer
        self._update_params_frequency = update_params_frequency
        self._accumulate_sum_or_mean = accumulate_sum_or_mean
        self._ema_decay = ema_decay
        self._set_hyper('update_params_frequency', self._update_params_frequency)
        self._set_hyper('ema_decay', self._ema_decay)

        self._num_of_aggregations = self.add_weight("num_of_aggregations", shape=[], dtype=tf.dtypes.int32, initializer="ones", trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self._weights.append(self._num_of_aggregations)

    def set_weights_ema(self):
        vars_ema = [self.get_slot(var, "var_ema") for var in self.var_list_for_ema_tracking]
        return tf.group(*[K.update(p, e_p / (1 - K.pow(self._ema_decay, tf.cast(self._optimizer.iterations, tf.float32)))) for e_p, p in zip(vars_ema, self.var_list_for_ema_tracking)])
    
    def apply_gradients(self, grads_and_vars, name=None):
        self._update_flag = tf.equal(tf.math.floormod(self._num_of_aggregations, self._get_hyper('update_params_frequency', tf.dtypes.int32)), 0)
        def update_fn():
            with tf.control_dependencies([tf.compat.v1.assign_add(self._optimizer.iterations, 1)]):
                return tf.compat.v1.assign(self._num_of_aggregations, 1)
        def just_store_fn():
            return tf.compat.v1.assign_add(self._num_of_aggregations, 1)
        with tf.control_dependencies([tf.cond(self._update_flag, update_fn, just_store_fn)]):
            return super(AccumulateGradientOptimizerWrapper, self).apply_gradients(grads_and_vars, name)

    def _resource_apply(self, update_func, grad, var, apply_state=None):
        accumulated_grad_op = self.get_slot(var, "accumulated_grad").assign_add(grad)
        with tf.control_dependencies([accumulated_grad_op]):
            def update_fn():
                if self._accumulate_sum_or_mean:
                    grad = self.get_slot(var, "accumulated_grad")
                else:
                    grad = self.get_slot(var, "accumulated_grad") / self._get_hyper('update_params_frequency', var.dtype)
                update_params_ops = update_func(grad, var, apply_state=apply_state)
                with tf.control_dependencies([update_params_ops]):
                    update_op = [self.get_slot(var, "accumulated_grad").assign(tf.zeros_like(var))]
                    if self._ema_decay > 0:
                        var_ema = self.get_slot(var, "var_ema")
                        update_op += [K.update(var_ema, (var_ema * self._ema_decay) + (1 - self._ema_decay) * var)]
                return tf.group(*[update_op])
            def just_store_fn():
                return tf.group(*[tf.no_op()])
            return tf.cond(self._update_flag, update_fn, just_store_fn)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        update_func = self._optimizer._resource_apply_dense
        return self._resource_apply(update_func, grad, var, apply_state=apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        def update_func(grad, var, apply_state=None):
            return self._optimizer._resource_apply_sparse(grad, var, indices, apply_state=apply_state)
        return self._resource_apply(update_func, grad, var, apply_state=apply_state)

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "accumulated_grad")
            if self._ema_decay > 0:
                self.add_slot(var, "var_ema")
        if self._ema_decay > 0:
            self.var_list_for_ema_tracking = var_list

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _create_all_weights(self, var_list):
        _ = self._optimizer.iterations
        return super()._create_all_weights(var_list)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            "accumulate_sum_or_mean": self._accumulate_sum_or_mean,
            'update_params_frequency': self._serialize_hyperparameter('update_params_frequency'),
            'ema_decay': self._serialize_hyperparameter('ema_decay'),
        }
        base_config = super().get_config()
        return {**base_config, **config}


    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects,
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)
    
    def set_update_params_frequency(self, frequency):
        self._set_hyper("update_params_frequency", frequency)
        print('setting "update_params_frequency" to %d results in %d' % (frequency, self._get_hyper('update_params_frequency', tf.dtypes.int32)))
def convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=True, ema_decay=0):
    return AccumulateGradientOptimizerWrapper(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=accumulate_sum_or_mean, ema_decay=ema_decay)

