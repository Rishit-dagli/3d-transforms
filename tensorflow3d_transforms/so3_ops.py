import math
from typing import Tuple

import tensorflow as tf

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def acos_linear_extrapolation(
    x: tf.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> tf.Tensor:
    """Implements `arccos(x)` which is linearly extrapolated outside `x`'s
    original domain of `(-1, 1)`. This allows for stable backpropagation in
    case `x` is not guaranteed to be strictly within `(-1, 1)`.

    More specifically:

    .. code-block:: python

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Example:

    .. code-block:: python

        x = tf.constant((1.0, 1.0, 1.0, 1.0, 1.0))
        acos_linear_extrapolation(x=x)
        # <tf.Tensor: shape=(5,), dtype=float32, numpy=
        # array([0.00706984, 0.00706984, 0.00706984, 0.00706984, 0.00706984],
        #       dtype=float32)>

    Args:
        x (tf.Tensor): Input `Tensor`.
        bounds (Tuple[float, float]): A float 2-tuple defining the
            region for the linear extrapolation of `acos`. The
            first/second element of `bound` describes the lower/upper
            bound that defines the lower/upper extrapolation region,
            i.e. the region where `x <= bound[0]`/`bound[1] <= x`. Note
            that all elements of `bound` have to be within (-1, 1).

    Returns:
        tf.Tensor: acos_linear_extrapolation: `Tensor` containing the
        extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    acos_extrap = tf.experimental.numpy.empty(x.shape, dtype=x.dtype)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    acos_extrap = tf.where(x_mid, tf.math.acos(x), acos_extrap)
    acos_extrap = tf.where(
        x_upper, _acos_linear_approximation(x, upper_bound), acos_extrap
    )
    acos_extrap = tf.where(
        x_lower, _acos_linear_approximation(x, lower_bound), acos_extrap
    )

    return acos_extrap


def _acos_linear_approximation(x: tf.Tensor, x0: float) -> tf.Tensor:
    """Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`."""
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """Calculates the derivative of `arccos(x)` w.r.t. `x`."""
    return (-1.0) / math.sqrt(1.0 - x * x)
