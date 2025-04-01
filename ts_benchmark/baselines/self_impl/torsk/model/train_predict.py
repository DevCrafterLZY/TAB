import logging

import numpy as np

from ts_benchmark.baselines.self_impl.torsk.model.numpy_accelerate import bh

logger = logging.getLogger(__name__)


def initial_state(hidden_size, dtype, backend):
    if backend == "numpy":
        zero_state = bh.zeros([hidden_size], dtype=np.float64)
    else:
        raise ValueError(f"Unkown backend: {backend}")
    return zero_state


def train_esn(model, dataset):
    inputs, labels, _ = dataset[0]

    tlen = model.params.transient_length
    hidden_size = model.esn_cell.hidden_size
    backend = model.params.backend
    dtype = model.esn_cell.dtype

    zero_state = initial_state(hidden_size, dtype, backend)
    _, states = model.forward(inputs, zero_state, states_only=True)

    logger.info("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    return inputs, states, labels


def train_predict_esn(model, dataset, steps=1, step_length=1, step_start=0):
    tlen = model.params.transient_length
    hidden_size = model.esn_cell.hidden_size
    backend = model.params.backend
    dtype = model.esn_cell.dtype

    predictions = np.empty((steps, model.params.pred_length, model.params.input_shape[0], model.params.input_shape[1]))
    targets = np.empty((steps, model.params.pred_length, model.params.input_shape[0], model.params.input_shape[1]))

    for ii in range(steps):
        model.timer.reset()

        idx = ii * step_length + step_start
        inputs, labels, pred_targets = dataset[idx]

        logger.debug(f"Creating {inputs.shape[0]} training states")
        zero_state = initial_state(hidden_size, dtype, backend)
        _, states = model.forward(inputs, zero_state, states_only=True)

        logger.debug("Optimizing output weights")
        model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

        logger.debug(f"Predicting the next {model.params.pred_length} frames")
        init_inputs = labels[-1]
        outputs, out_states = model.predict(
            init_inputs, states[-1], nr_predictions=model.params.pred_length)

        logger.debug(model.timer.pretty_print())

        predictions[ii, :, :, :] = outputs
        targets[ii, :, :, :] = pred_targets

    logger.info(f"Done")
    return predictions, targets
