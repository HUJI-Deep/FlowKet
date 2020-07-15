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
from tensorflow.keras.initializers import Initializer


class CopyNormaInitializer(Initializer):
    """docstring for NegateDecorator"""

    def __init__(self, variable, exponential_norm=False):
        super(CopyNormaInitializer, self).__init__()
        self.variable = variable
        self.exponential_norm = exponential_norm

    def __call__(self, shape, dtype=None, partition_info=None):
        flat = tf.reshape(self.variable.initial_value, [-1, shape[-1]])
        norm = tf.linalg.norm(flat, axis=0)
        if self.exponential_norm:
            norm = tf.math.log(norm + 1e-10)
        return tf.reshape(norm, shape)


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

    def __init__(self, layer, normalize_per_output_channel=True,
                 exponential_norm=False, **kwargs):
        super(WeightNormalization, self).__init__(layer, **kwargs)
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
            self.layer.g = self.layer.add_weight(
                name="g",
                shape=(self.layer_depth,),
                initializer=CopyNormaInitializer(self.v, exponential_norm=self.exponential_norm),
                dtype=self.layer.kernel.dtype,
                trainable=True)
            self.g = self.layer.g
            self.layer.g = None # hack for removing g from self.layer.trainable_weights in a way that work both  fr=or td 1.10 & 1.14
        super(WeightNormalization, self).build()

    def call(self, inputs):
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
                g = tf.math.exp(g)
            self.layer.kernel = tf.nn.l2_normalize(
                self.v, axis=self.kernel_norm_axes) * g

    def get_config(self):
        config = {'normalize_per_output_channel': self.normalize_per_output_channel,
                  'exponential_norm': self.exponential_norm}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
