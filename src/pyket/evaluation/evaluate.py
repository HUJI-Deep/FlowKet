import numpy
import tqdm


def mean_logs(logs_arr, keys=None):
    # assuming same keys at each epoch ...
    res = {}
    if keys is None:
        keys = logs_arr[0].keys()
    for key in keys:
        res[key] = numpy.mean([logs[key] for logs in logs_arr])
    return res

    
def evaluate(generator, steps, callbacks, keys_to_progress_bar_mapping=None):
    logs_arr = []
    with tqdm.trange(steps) as progress_bar:
        for i in progress_bar:
            next(generator)
            logs = {}
            for callback in callbacks:
                callback.on_batch_end(i, logs)
                callback.on_epoch_end(i, logs)
            logs_arr.append(logs)
            if keys_to_progress_bar_mapping is not None:
                to_show = mean_logs(logs_arr, keys=keys_to_progress_bar_mapping)
                progress_bar.set_postfix({key_to_show: to_show[key] for key, key_to_show in keys_to_progress_bar_mapping.items()})
    return mean_logs(logs_arr)


def exact_evaluate(exact_variational, callbacks):
    exact_variational.machine_updated()
    logs = {}
    for callback in callbacks:
        callback.on_batch_end(exact_variational.num_of_batch_until_full_cycle, logs)
        callback.on_epoch_end(1, logs)
    return logs