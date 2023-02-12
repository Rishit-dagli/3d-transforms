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
    """Calculates the derivative of `arccos(x)` w.r.t.

    `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


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
    """Calculates the derivative of `arccos(x)` w.r.t.

    `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


def so3_rotation_angle(
    R: tf.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> tf.Tensor:
    """Calculates angles (in radians) of a batch of rotation matrices `R` with.

    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Example:

    .. code-block:: python

        v = tf.constant([[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])
        so3_rotation_angle(v)
        # <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.00706984], dtype=float32)>

    Args:
        R (tf.Tensor): Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps (float): Tolerance for the valid trace check.
        cos_angle (bool): If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound (float): Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
    Returns:
        tf.Tensor: Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """
    _, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    if tf.math.reduce_any(rot_trace < -1.0 - eps) or tf.math.reduce_any(
        rot_trace > 3.0 + eps
    ):
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return tf.math.acos(phi_cos)


def hat(v: tf.Tensor) -> tf.Tensor:
    """Computes the hat operator of a batch of 3D vector.

    Example:

    .. code-block:: python

        v = tf.constant([[1., 1., 1.], [1., 1., 1.]])
        hat(v)
        # <tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
        # array([[[ 0., -1.,  1.],
        #         [ 1.,  0., -1.],
        #         [-1.,  1.,  0.]],

        #        [[ 0., -1.,  1.],
        #         [ 1.,  0., -1.],
        #         [-1.,  1.,  0.]]], dtype=float32)>

    Args:
        v (tf.Tensor): Batch of 3D vectors of shape `(minibatch, 3)`.
    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`
    Raises:
        ValueError if `v` is of incorrect shape.
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = tf.Variable(tf.zeros((N, 3, 3), dtype=v.dtype))

    x, y, z = tf.unstack(v, axis=1)

    h[:, 0, 1].assign(-z)
    h[:, 0, 2].assign(y)
    h[:, 1, 0].assign(z)
    h[:, 1, 2].assign(-x)
    h[:, 2, 0].assign(-y)
    h[:, 2, 1].assign(x)

    return tf.convert_to_tensor(h)


def hat_inverse(h: tf.Tensor) -> tf.Tensor:
    """Computes the inverse hat operator of a batch of skew-symmetric matrices.

    Example:

    .. code-block:: python

        h = tf.constant([[[ 0., -1.,  1.],
            [ 1.,  0., -1.],
            [-1.,  1.,  0.]],
        [[ 0., -1.,  1.],
            [ 1.,  0., -1.],
            [-1.,  1.,  0.]]])
        hat_inverse(h)
        # <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
        # array([[1., 1., 1.],
        #        [1., 1., 1.]], dtype=float32)>

    Args:
        h (tf.Tensor): Batch of skew-symmetric matrices of shape
            `(minibatch, 3, 3)`.

    Returns:
        Batch of 3D vectors of shape `(minibatch, 3)` where each vector is of the form:

    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` is not skew-symmetric.
    """
    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = tf.reduce_max(tf.abs(h + tf.transpose(h, perm=[0, 2, 1])))

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = tf.stack((x, y, z), axis=1)

    return v


def _so3_exp_map(
    log_rot: tf.Tensor, eps: float = 0.0001
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Computes the exponential map of a batch of 3D rotation vectors and
    returns the intermediate tensors as well.

    Example:

    .. code-block:: python

    log_rot = tf.constant(((1.,1.,1.),(1.,1.,1.)))
    _so3_exp_map(log_rot)
    # (<tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
    #  array([[[ 0.22629565, -0.18300793,  0.95671225],
    #          [ 0.95671225,  0.22629565, -0.18300793],
    #          [-0.18300793,  0.95671225,  0.22629565]],
    #         [[ 0.22629565, -0.18300793,  0.95671225],
    #          [ 0.95671225,  0.22629565, -0.18300793],
    #          [-0.18300793,  0.95671225,  0.22629565]]], dtype=float32)>,
    # 
    #  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.7320508, 1.7320508], dtype=float32)>,
    # 
    #  <tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
    #  array([[[ 0., -1.,  1.],
    #          [ 1.,  0., -1.],
    #          [-1.,  1.,  0.]],
    #         [[ 0., -1.,  1.],
    #          [ 1.,  0., -1.],
    #          [-1.,  1.,  0.]]], dtype=float32)>,
    # 
    #  <tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
    #  array([[[-2.,  1.,  1.],
    #          [ 1., -2.,  1.],
    #          [ 1.,  1., -2.]],
    #         [[-2.,  1.,  1.],
    #          [ 1., -2.,  1.],
    #          [ 1.,  1., -2.]]], dtype=float32)>)

    Args:
        log_rot (tf.Tensor): Batch of 3D rotation vectors of shape
            `(minibatch, 3)`.
        eps (float): Threshold for the norm of the rotation vector.
            If the norm is below this threshold, the exponential map
            is approximated with the first order Taylor expansion.

    Returns:
        Tuple of tf.Tensor: Batch of rotation matrices of shape
            `(minibatch, 3, 3)`, batch of rotation angles of shape
            `(minibatch,)`, batch of rotation vectors of shape
            `(minibatch, 3)`, batch of rotation vector norms of shape
            `(minibatch,)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.
    """
    _, dim = tf.shape(log_rot)
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = tf.reduce_sum(log_rot * log_rot, axis=1)
    rot_angles = tf.clip_by_value(nrms, eps, tf.math.reduce_max(nrms))**0.5
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * tf.math.sin(rot_angles)
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - tf.math.cos(rot_angles))
    skews = hat(log_rot)
    skews_square = tf.matmul(skews, skews)

    eye_3 = tf.eye(3, dtype=log_rot.dtype)
    R = fac1[:, tf.newaxis, tf.newaxis] * skews + fac2[:, tf.newaxis, tf.newaxis] * skews_square + tf.tile(tf.expand_dims(eye_3, 0), [tf.shape(log_rot)[0], 1, 1])

    return R, rot_angles, skews, skews_square


def so3_exp_map(log_rot: tf.Tensor, eps: float = 0.0001) -> tf.Tensor:
    """Computes the exponential map of a batch of 3D rotation vectors.

    Example:

    .. code-block:: python

        log_rot = tf.constant(((1.,1.,1.),(1.,1.,1.)))
        so3_exp_map(log_rot)
        # <tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
        # array([[[ 0.22629565, -0.18300793,  0.95671225],
        #         [ 0.95671225,  0.22629565, -0.18300793],
        #         [-0.18300793,  0.95671225,  0.22629565]],
        #        [[ 0.22629565, -0.18300793,  0.95671225],
        #         [ 0.95671225,  0.22629565, -0.18300793],
        #         [-0.18300793,  0.95671225,  0.22629565]]], dtype=float32)>

    Args:
        log_rot (tf.Tensor): Batch of 3D rotation vectors of shape
            `(minibatch, 3)`.
        eps (float): Threshold for the norm of the rotation vector.
            If the norm is below this threshold, the exponential map
            is approximated with the first order Taylor expansion.

    Returns:
        tf.Tensor: Batch of rotation matrices of shape
            `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.
    """
    R, _, _, _ = _so3_exp_map(log_rot, eps)

    return R


def so3_log_map(
    R: tf.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4
) -> tf.Tensor:
    """Convert a batch of 3x3 rotation matrices `R` to a batch of 3-dimensional
    matrix logarithms of rotation matrices The conversion has a singularity
    around `(R=I)` which is handled by clamping controlled with the `eps` and
    `cos_bound` arguments.

    Example:

    .. code-block:: python

        R = tf.constant([[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]])
        so3_log_map(R)
        # <tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[0., 0., 0.]], dtype=float32)>

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call when computing `so3_rotation_angle`.
            Note that the non-finite outputs/gradients are returned when
            the rotation angle is close to 0 or π.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    _, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = tf.math.sin(phi)
    phi_factor = tf.zeros_like(phi)
    ok_denom = tf.math.abs(phi_sin) > (0.5 * eps)
    phi_factor = tf.where(ok_denom, phi / (2.0 * phi_sin), 0.5 + (phi ** 2) * (1.0 / 12))

    log_rot_hat = phi_factor[:, tf.newaxis, tf.newaxis] * (R - tf.transpose(R, [0, 2, 1]))

    log_rot = hat_inverse(log_rot_hat)

    return log_rot


def so3_relative_angle(
    R1: tf.Tensor,
    R2: tf.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> tf.Tensor:
    """Calculates the relative angle (in radians) between pairs of.

    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Example:

    .. code-block:: python

        R = tf.constant([[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])
        so3_relative_angle(R, R, eps=1e2)
        # <tf.Tensor: shape=(1,), dtype=float32, numpy=array([-212.13028], dtype=float32)>

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = tf.matmul(R1, tf.transpose(R2, [0, 2, 1]))
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)
