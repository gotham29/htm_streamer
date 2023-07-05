from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from htm.bindings.sdr import SDR


def sdr_max_pool(input_sdr: SDR, ratio: int, axis: int = 0) -> SDR:
    if ratio > 1:
        dims = input_sdr.dimensions
        dims[axis] //= ratio

        new_coords = np.array(input_sdr.coordinates)
        new_coords[axis] //= ratio

        new_sdr = SDR(dims)
        new_sdr.coordinates = new_coords

        return new_sdr
    else:
        return input_sdr


def _check_shapes(*inputs) -> np.ndarray:
    s0 = inputs[0].dimensions
    for idx, sdr in enumerate(inputs):
        if s0 != sdr.dimensions:
            raise RuntimeError(f"All SDRs must have the same dimensions, got {s0} for sdr_0 and {sdr.dimensions} for "
                               f"sdr_{idx}")
    return s0


def sdr_merge(*inputs, mode: str, axis: int = 0) -> SDR:
    if len(inputs) == 1:
        return inputs[0]

    shape = _check_shapes(*inputs)

    # union
    if mode == 'u':
        return SDR(shape).union(inputs)

    # intersection
    elif mode == 'i':
        return SDR(shape).intersection(inputs)

    # symmetric difference
    elif mode == 'sd' or mode == 'xor':
        return sdr_symmetric_diff(*inputs)

    # concat
    elif mode == 'c':
        old_shapes = [sdr.shape for sdr in inputs]
        new_shape = concat_shapes(*old_shapes, axis=axis)
        return SDR(new_shape).concatenate(inputs, axis=axis)

    else:
        raise ValueError(f"Unknown SDR merger mode `{mode}`")


def sdr_symmetric_diff(*inputs) -> SDR:
    shape = inputs[0].dimensions
    return sdr_subtract(SDR(shape).union(inputs), SDR(shape).intersection(inputs))


def sdr_subtract(sdr_1: SDR, sdr_2: SDR) -> SDR:
    if sdr_1.dimensions != sdr_2.dimensions:
        raise ValueError(f"When subtracting, both SDRs must have the same dimensions, got: `{sdr_1.dimensions}`,"
                         f" `{sdr_2.dimensions}`")

    new_sdr = SDR(sdr_1.dimensions)
    new_sdr.sparse = np.setdiff1d(sdr_1.sparse, sdr_2.sparse)  # get diff with sets

    return new_sdr


def concat_shapes(*shapes, axis=0) -> tuple:
    if len(shapes) < 2:
        raise ValueError("Cannot concatenate less than 2 shapes")

    if len(set(len(s) for s in shapes)) != 1:
        raise ValueError("All shapes must have same number of dimensions")

    ndim = len(shapes[0])
    if axis < 0:
        _old = axis
        axis = ndim + _old
        if axis < 0:
            raise ValueError(f"Invalid axis {_old} for shapes with {ndim} dimensions")

    if axis >= ndim:
        raise ValueError(f"Invalid axis {axis} for shapes with {ndim} dimensions")

    for dim, _ in enumerate(shapes[0]):
        if dim == axis:
            continue
        if len(set(s[dim] for s in shapes)) != 1:
            raise ValueError(
                f"Shapes must be equal in all axes except the concatenated axis, got different shapes for axis {dim}")

    new_shape = []
    for dim, size in enumerate(shapes[0]):
        if dim == axis:
            size = sum(s[dim] for s in shapes)
        new_shape.append(size)

    return tuple(new_shape)


def flatten_shape(shape: np.ndarray) -> np.ndarray:
    return np.prod(shape, keepdims=True)


def squeeze_shape(shape: List[int] | Tuple[int]) -> Tuple[int]:
    """ Squeezes the shape by merging last 2 dims """
    if len(shape) < 2:
        raise ValueError("Cannot squeeze shape with less than 2 dims")
    else:
        new_last = shape[-1] * shape[-2]
        new_shape = [dim for dim in shape[:-2]]
        new_shape.append(new_last)
        return tuple(new_shape)
