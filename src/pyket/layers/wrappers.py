# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# based on from https://github.com/tensorflow/addons
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class WeightNormalization(tf.keras.layers.Wrapper):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.

    This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNormalization wrapper works for keras and tf layers.
    ```python
      net = WeightNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = WeightNormalization(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """

    def __init__(self, layer, data_init=True, normalize_per_output_channel=True, 
                 exponential_norm=False, **kwargs):
        super(WeightNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self.normalize_per_output_channel = normalize_per_output_channel
        self.exponential_norm = exponential_norm

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('`WeightNormalization` must wrap a layer that'
                                 ' contains a `kernel` for weights')

            # The kernel's filter or unit dimension is -1
            if self.normalize_per_output_channel:
                self.layer_depth = int(self.layer.kernel.shape[-1])
                self.kernel_norm_axes = list(
                    range(len(self.layer.kernel.shape) - 1))
            else:
                self.layer_depth = 1
                self.kernel_norm_axes = list(
                    range(len(self.layer.kernel.shape)))
            self.v = self.layer.kernel
            g_init = 'zeros' if self.exponential_norm else 'ones'
            self.g = self.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=tf.keras.initializers.get(g_init),
                dtype=self.layer.kernel.dtype,
                trainable=True)
        self._initialized = self.add_variable(
                name="initialized",
                shape=(),
                initializer=tf.keras.initializers.get('zeros'),
                dtype=tf.bool,
                trainable=False)

        super(WeightNormalization, self).build()

    def call(self, inputs):
        """Call `Layer`"""
        # todo more effiect implementation ?
        init_op = tf.cond(self._initialized, true_fn=lambda :tf.group(*[tf.no_op()]), false_fn=lambda :self._initialize_weights(tf.stop_gradient(inputs)))
        with tf.control_dependencies([init_op]):
            self._compute_weights()  # Recompute weights for each forward pass
            output = self.layer(inputs)
            return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _compute_weights(self):
        """Generate normalized weights.

        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        with tf.name_scope('compute_weights'):
            g = self.g
            if self.exponential_norm:
                g = tf.exp(g)
            self.layer.kernel = tf.nn.l2_normalize(
                self.v, axis=self.kernel_norm_axes) * g

    def _initialize_weights(self, inputs):
        """Initialize weight g.

        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        if self.data_init:
            init_ops = self._data_dep_init(inputs)
        else:
            init_ops = self._init_norm()
        init_ops.append(self._initialized.assign(np.ones(shape=(), dtype=np.bool)))
        return tf.group(*init_ops)

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope('init_norm'):
            flat = tf.reshape(self.v, [-1, self.layer_depth])
            norm = tf.linalg.norm(flat, axis=0)
            if self.exponential_norm:
                norm = tf.log(norm + 1e-10)
            return [self.g.assign(
                tf.reshape(norm, (self.layer_depth,)))]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""

        with tf.name_scope('data_dep_init'):
            # Generate data dependent init values
            existing_activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer(inputs)
            if self.normalize_per_output_channel:
                data_norm_axes = list(range(len(x_init.shape) - 1))
            else:
                data_norm_axes = list(range(len(x_init.shape)))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.sqrt(v_init + 1e-10)

        # Assign data dependent init values
        init_ops = []

        if self.exponential_norm:
            init_ops.append(tf.assign_add(self.g, tf.log(scale_init + 1e-10)))
        else:
            init_ops.append(tf.assign(self.g, self.g * scale_init))
        if hasattr(self.layer, 'bias'):
            init_ops.append(tf.assign(self.layer.bias, -m_init * scale_init))
        self.layer.activation = existing_activation
        return init_ops

    def get_config(self):
        config = {'data_init': self.data_init, 
                  'normalize_per_output_channel': self.normalize_per_output_channel, 
                  'exponential_norm' : self.exponential_norm}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
