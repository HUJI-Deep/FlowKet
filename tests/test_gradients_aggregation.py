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
    optimizer = convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency=update_params_frequency,
                                             accumulate_sum_or_mean=accumulate_sum_or_mean, ema_decay=ema_decay)

    def y_loss(y_true, y_pred):
        return K.mean(y_pred)

    def get_w():
        return model.get_weights()[0][0][0].item()

    def get_sgd_iteration():
        iteration_weight_index = None
        for i, w in enumerate(optimizer.weights):
            if w.name == orig_optimizer.iterations.name:
                iteration_weight_index = i
        return optimizer.get_weights()[iteration_weight_index].item()

    def set_weights_ema():
        optimizer.set_weights_ema()

    def set_update_params_frequency(frequency):
        optimizer.set_update_params_frequency(frequency)
        model.compile(optimizer=optimizer, loss=y_loss)

    model.compile(optimizer=optimizer, loss=y_loss)
    return model, get_w, get_sgd_iteration, set_weights_ema, set_update_params_frequency


def test_update_just_when_need():
    model, get_w, get_sgd_iteration, _, _ = get_simple_linear_model(SGD(lr=1.0), 2, False)
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
    model, get_w, get_sgd_iteration, _, set_update_params_frequency = get_simple_linear_model(optimizer, 1, False)
    set_update_params_frequency(2)
    w_before_call = get_w()
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_first_call = get_w()
    global_step_after_first_call = get_sgd_iteration()
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    global_step_after_second_call = get_sgd_iteration()
    set_update_params_frequency(1)
    model.fit(x=np.array([[1.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    global_step_after_third_call = get_sgd_iteration()
    w_after_third_call = get_w()
    assert global_step_after_first_call == 0
    assert global_step_after_second_call == 1
    assert global_step_after_third_call == 2
    assert w_before_call == 1.0
    assert w_after_first_call == 1.0
    assert w_after_second_call == -1.5
    assert w_after_third_call == -2.5


def test_reset_after_update():
    model, get_w, get_sgd_iteration, _, _ = get_simple_linear_model(SGD(lr=1.0), 1, False)
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    assert w_after_second_call == -4.0


def test_update_ema_just_when_need():
    model, get_w, get_sgd_iteration, set_weights_ema, _ = get_simple_linear_model(SGD(lr=1.0), 2, False, .9)
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
    assert w_after_set_ema == pytest.approx(-1.5)


def test_ema():
    model, get_w, get_sgd_iteration, set_weights_ema, _ = get_simple_linear_model(SGD(lr=1.0), 1, False, .9)
    model.fit(x=np.array([[2.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_first_call = get_w()
    model.fit(x=np.array([[3.0]], dtype=np.float32), y=np.array([[0.0]], dtype=np.float32), batch_size=1)
    w_after_second_call = get_w()
    set_weights_ema()
    w_after_set_ema = get_w()
    assert w_after_first_call == -1.0
    assert w_after_second_call == -4.0
    assert w_after_set_ema == pytest.approx(-2.5789473684210527)
