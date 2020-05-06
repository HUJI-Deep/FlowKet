from flowket.optimizers import convert_to_accumulate_gradient_optimizer

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

import numpy as np
import pytest
import tensorflow as tf


def get_simple_linear_model(orig_optimizer, update_params_frequency, accumulate_sum_or_mean, ema_decay=0):
    inputs = Input(shape=(1,), dtype='float32')
    outputs = Dense(1, use_bias=False, kernel_initializer='ones')(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency=update_params_frequency,
                                             accumulate_sum_or_mean=accumulate_sum_or_mean, ema_decay=ema_decay)

    def y_loss(y_true, y_pred):
        return K.mean(y_pred)

    def get_w():
        return model.get_weights()[0][0][0].item()

    def get_sgd_iteration():
        return orig_optimizer.get_weights()[orig_optimizer.weights.index(orig_optimizer.iterations)].item()

    def set_weights_ema():
        orig_optimizer.set_weights_ema()

    model.compile(optimizer=orig_optimizer, loss=y_loss)
    return model, get_w, get_sgd_iteration, set_weights_ema


def test_update_just_when_need():
    model, get_w, get_sgd_iteration, _ = get_simple_linear_model(SGD(lr=1.0), 2, False)
    w_before_call = get_w()
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_first_call = get_w()
    global_step_after_first_call = get_sgd_iteration()
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    global_step_after_second_call = get_sgd_iteration()
    assert global_step_after_first_call == 0
    assert global_step_after_second_call == 1
    assert w_before_call == 1.0
    assert w_after_first_call == 1.0
    assert w_after_second_call == -1.5


def test_changing_the_update_frequency():
    optimizer = SGD(lr=1.0)
    model, get_w, get_sgd_iteration, _ = get_simple_linear_model(optimizer, 1, False)
    optimizer.set_update_params_frequency(2)
    w_before_call = get_w()
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_first_call = get_w()
    global_step_after_first_call = get_sgd_iteration()
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    global_step_after_second_call = get_sgd_iteration()
    optimizer.set_update_params_frequency(1)
    model.fit(x=np.array([[1.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_third_call = get_w()
    assert global_step_after_first_call == 0
    assert global_step_after_second_call == 1
    assert w_before_call == 1.0
    assert w_after_first_call == 1.0
    assert w_after_second_call == -1.5
    assert w_after_third_call == -2.5


def test_reset_after_update():
    model, get_w, get_sgd_iteration, _ = get_simple_linear_model(SGD(lr=1.0), 1, False)
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    assert w_after_second_call == -4.0


def test_update_ema_just_when_need():
    model, get_w, get_sgd_iteration, set_weights_ema = get_simple_linear_model(SGD(lr=1.0), 2, False, .9)
    w_before_call = get_w()
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_first_call = get_w()
    global_step_after_first_call = get_sgd_iteration()
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    set_weights_ema()
    w_after_set_ema = get_w()
    global_step_after_second_call = get_sgd_iteration()
    assert global_step_after_first_call == 0
    assert global_step_after_second_call == 1
    assert w_before_call == 1.0
    assert w_after_first_call == 1.0
    assert w_after_second_call == -1.5
    assert w_after_set_ema == -1.5


def test_ema():
    model, get_w, get_sgd_iteration, set_weights_ema = get_simple_linear_model(SGD(lr=1.0), 1, False, .9)
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    set_weights_ema()
    w_after_set_ema = get_w()
    assert w_after_first_call == -1.0
    assert w_after_second_call == -4.0
    assert w_after_set_ema == pytest.approx(-2.5789)