"""This module contains functions to convert between rotation representations.

The transformation matrices returned from the functions in this file
assume the points on which the transformation will be applied are column
vectors that is the R matrix is structured as

.. code-block:: python

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

Furthermore, we will assume for any functions in this module that these are
quaternions with real part first that is a tensor of shape (..., 4).
"""

from typing import Optional

import tensorflow as tf


def quaternion_to_matrix(quaternions: tf.Tensor) -> tf.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Example:

    .. code-block:: python

        quaternion = tf.constant([0.0, 0.0, 0.0, 4.0])
        output = tensorflow3d_transforms.quaternion_to_matrix(quaternions=quaternion)
        # <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
        # array([[-1.,  0.,  0.],
        #     [ 0., -1.,  0.],
        #     [ 0.,  0.,  1.]], dtype=float32)>

    Args:
        quaternions (tf.Tensor): A tensor of shape (..., 4) representing
            quaternions with real part first.

    Returns:
        tf.Tensor: A tensor of shape (..., 3, 3) representing rotation
        matrices.
    """
    r, i, j, k = tf.unstack(quaternions, axis=-1)
    two_s = 2.0 / tf.reduce_sum(quaternions * quaternions, axis=-1)

    o = tf.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        -1,
    )
    return tf.reshape(o, quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: tf.Tensor) -> tf.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Example:

    .. code-block:: python

        matrix = tf.constant(
            [
                [
                    [0.15885946, -0.56794965, -0.48926896],
                    [-1.0064808, -0.39120296, 1.6047943],
                    [0.05503756, 0.817741, 0.4543775],
                ]
            ]
        )

        matrix_to_quaternion(matrix)
        # <tf.Tensor: shape=(1, 4), dtype=float32, numpy=
        # array([[-0.1688297 , -0.16717434,  0.9326495 ,  0.6493691 ]],
        # dtype=float32)>

    Args:
        matrix (tf.Tensor): A tensor of shape (..., 3, 3) representing
            rotation matrices.

    Returns:
        tf.Tensor: A tensor of shape (..., 4) representing quaternions
        with real part first.

    Raises:
        ValueError: If the shape of the input matrix is invalid that is
            does not have the shape (..., 3, 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = tf.unstack(
        tf.reshape(matrix, batch_dim + (9,)), axis=-1
    )

    q_abs = _sqrt_positive_part(
        tf.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    quat_by_rijk = tf.stack(
        [
            tf.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            tf.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            tf.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            tf.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    flr = tf.convert_to_tensor(0, dtype=tf.int32)
    quat_candidates = quat_by_rijk / (
        2.0 * tf.reduce_max(q_abs[..., None], flr, keepdims=True)
    )
    max_indices = tf.argmax(q_abs, axis=-1)
    one_hot = tf.one_hot(max_indices, depth=4)
    selected = tf.boolean_mask(quat_candidates, one_hot > 0.5)
    return tf.reshape(selected, batch_dim + [4])


def _sqrt_positive_part(x: tf.Tensor) -> tf.Tensor:
    """Returns the square root of all positive elements of x and 0 for others.

    Args:
        x (tf.Tensor): A tensor

    Returns:
        tf.Tensor: A tensor with the same shape as x
    """
    ret = tf.zeros_like(x)
    positive_mask = x > 0
    ret = tf.where(positive_mask, tf.math.sqrt(x), ret)
    return ret


def _axis_angle_rotation(axis: str, angle: tf.Tensor) -> tf.Tensor:
    """Return the rotation matrices for one of the rotations about an axis of
    which Euler angles describe, for each value of the angle given.

    Args:
        axis (str): The axis about which the rotation is performed. Must
            be one of 'X', 'Y', 'Z'.
        angle (tf.Tensor): Any shape tensor of Euler angles in radians

    Returns:
        tf.Tensor: A tensor of shape (..., 3, 3) representing rotation
        matrices.

    Raises:
        ValueError: If the axis is not one of 'X', 'Y', 'Z'.
    """
    if axis not in ("X", "Y", "Z"):
        raise ValueError("letter must be either X, Y or Z.")

    cos = tf.math.cos(angle)
    sin = tf.math.sin(angle)
    one = tf.ones_like(angle)
    zero = tf.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return tf.reshape(tf.stack(R_flat, axis=-1), angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: tf.Tensor, convention: str) -> tf.Tensor:
    """Convert rotations given as euler angles to rotation matrices.

    Example:

    .. code-block:: python

        euler_angles = tf.constant(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )

        euler_angles_to_matrix(euler_angles=euler_angles, convention="XYZ")
        # <tf.Tensor: shape=(1, 3, 3, 3), dtype=float32, numpy=
        # array([[[[1., 0., 0.],
        #          [0., 1., 0.],
        #          [0., 0., 1.]],
        #
        #         [[1., 0., 0.],
        #          [0., 1., 0.],
        #          [0., 0., 1.]],
        #
        #         [[1., 0., 0.],
        #          [0., 1., 0.],
        #          [0., 0., 1.]]]], dtype=float32)>

    Args:
        euler_angles (tf.Tensor): A tensor of shape (..., 3)
            representing euler angles.
        convention (str): The euler angle convention. A string
            containing a combination of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        tf.Tensor: A tensor of shape (..., 3, 3) representing rotation
        matrices.

    Raises:
        ValueError: If the shape of the input euler angles is invalid
            that is does not have the shape (..., 3).
        ValueError: If the convention string is invalid that is does not
            have the length 3.
        ValueError: If the second character of the convention string is
            the same as the first or third.
        ValueError: If the convention string contains characters other
            than {"X", "Y", and "Z"}.
    """

    if euler_angles.shape[-1] != 3:
        raise ValueError(
            f"Invalid euler angle shape {euler_angles.shape}, last dimension should"
            " be 3."
        )
    if len(convention) != 3:
        raise ValueError(
            f"Invalid euler angle convention {convention}, should be a string of"
            " length 3."
        )
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(
            f"Invalid euler angle convention {convention}, second character should be"
            " different from first and third."
        )
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")

    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, tf.unstack(euler_angles, axis=-1))
    ]

    return tf.linalg.matmul(tf.linalg.matmul(matrices[0], matrices[1]), matrices[2])


def _index_from_letter(letter: str) -> int:
    """Return the index of the axis corresponding to the letter.

    Args:
        letter (str): The letter corresponding to the axis. Must be one
            of 'X', 'Y', 'Z'.

    Returns:
        int: The index of the axis.

    Raises:
        ValueError: If the letter is not one of 'X', 'Y', 'Z'.
    """
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data: tf.Tensor, horizontal: bool, tait_bryan: bool
) -> tf.Tensor:
    """Extract the first or third Euler angle from the two members of the
    matrix which are positive constant times its sine and cosine.

    Args:
        axis (str): Axis label "X" or "Y or "Z" for the angle we are
            finding.
        other_axis (str): Axis label "X" or "Y or "Z" for the middle
            axis in the convention.
        data (tf.Tensor): Rotation matrices as tensor of shape (..., 3,
            3).
        horizontal (bool): Whether we are looking for the angle for the
            third axis, which means the relevant entries are in the same
            row of the rotation matrix. If not, they are in the same
            column.
        tait_bryan (bool): Whether the first and third axes in the
            convention differ.

    Returns:
        tf.Tensor: Euler Angles in radians for each matrix in data as a
        tensor of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return tf.math.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return tf.math.atan2(-data[..., i2], data[..., i1])
    return tf.math.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: tf.Tensor, convention: str) -> tf.Tensor:
    """Convert rotation matrices to euler angles in radians.

    Example:

    .. code-block:: python

        matrix = tf.constant(
            [
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ]
        )
        matrix_to_euler_angles(matrix=matrix, convention="XYZ")
        # <tf.Tensor: shape=(1, 3, 3), dtype=float32, numpy=
        # array([[[-0.,  0., -0.],
        #         [-0.,  0., -0.],
        #         [-0.,  0., -0.]]], dtype=float32)>

    Args:
        matrix (tf.Tensor): A tensor of shape (..., 3, 3) representing
            rotation matrices.
        convention (str): The euler angle convention. A string
            containing a combination of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        tf.Tensor: A tensor of shape (..., 3) representing euler angles.

    Raises:
        ValueError: If the shape of the input matrix is invalid that is
            does not have the shape (..., 3, 3).
        ValueError: If the convention string is invalid that is does not
            have the length 3.
        ValueError: If the second character of the convention string is
            the same as the first or third.
        ValueError: If the convention string contains characters other
            than {"X", "Y", and "Z"}.
    """
    if len(convention) != 3:
        raise ValueError(
            f"Invalid euler angle convention {convention}, should be a string of"
            " length 3."
        )
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(
            f"Invalid euler angle convention {convention}, second character should be"
            " different from first and third."
        )
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Invalid matrix shape {matrix.shape}, last two dimensions should be 3, 3."
        )
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        if i0 - i2 in [-1, 2]:
            central_angle = tf.math.asin(-1 * matrix[..., i0, i2])
        else:
            central_angle = tf.math.asin(matrix[..., i0, i2])
    else:
        central_angle = tf.math.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return tf.stack(o, axis=-1)


def _copysign(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Return a tensor where each element has the absolute value taken from
    the, corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a (tf.Tensor): Source tensor.
        b (tf.Tensor): Tensor whose signs will be used, of the same
            shape as a.

    Returns:
        tf.Tensor: Tensor of the same shape as a with the signs of b.

    Raises:
        ValueError: If the shapes of a and b do not match.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shapes of a and b do not match: {a.shape} and {b.shape}.")
    signs_differ = (a < 0) != (b < 0)
    return tf.where(signs_differ, -a, a)


def random_quaternions(
    n: int,
    dtype: Optional[tf.dtypes.DType] = tf.float32,
) -> tf.Tensor:
    """Generate random quaternions representing rotations, i.e. versors with
    nonnegative real part.

    Example:

    .. code-block:: python

        random_quaternions(2)
        # <tf.Tensor: shape=(2, 4), dtype=float32, numpy=...>

    Args:
        n (int): Number of quaternions to generate.
        dtype (Optional[tf.dtype], optional): Data type of the returned
            tensor, defaults to tf.float32.

    Returns:
        tf.Tensor: Tensor of shape (n, 4) representing quaternions.
    """
    o = tf.random.normal((n, 4), dtype=dtype)
    s = tf.reduce_sum(o * o, axis=1)
    o = o / _copysign(tf.math.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int,
    dtype: Optional[tf.dtypes.DType] = tf.float32,
) -> tf.Tensor:
    """Generate random rotations as 3x3 rotation matrices.

    Example:

    .. code-block:: python

        random_rotations(2)
        # <tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=...>

    Args:
        n (int): Number of rotation matrices to generate.
        dtype (Optional[tf.dtype], optional): Data type of the returned
            tensor, defaults to tf.float32.

    Returns:
        tf.Tensor: Tensor of shape (n, 3, 3) representing rotation
        matrices.
    """
    quaternions = random_quaternions(n, dtype=dtype)
    return quaternion_to_matrix(quaternions)


def standardize_quaternion(quaternions: tf.Tensor) -> tf.Tensor:
    """Convert a unit quaternion to a standard form: one in which the real part
    is non negative.

    Example:

    .. code-block:: python

        quaternions = tf.constant((-1.,-2.,-1.,-1.))
        standardize_quaternion(quaternions=quaternions)
        # <tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 1.,  2.,  1.,  1.], dtype=float32)>

    Args:
        quaternions (tf.Tensor): Quaternions with real part first, as
            tensor of shape (..., 4).

    Returns:
        tf.Tensor: Standardized quaternions as tensor of shape (..., 4).
    """
    return tf.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Multiply two quaternions.

    Example:

    .. code-block:: python

        a = tf.constant((1.,2.,3.,4.))
        b = tf.constant((5.,6.,7.,8.))
        quaternion_raw_multiply(a=a, b=b)
        # <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-60.,  12.,  30.,  24.], dtype=float32)>

    Args:
        a (tf.Tensor): First quaternion with real part first, as tensor
            of shape (..., 4).
        b (tf.Tensor): Second quaternion with real part first, as tensor
            of shape (..., 4).

    Returns:
        tf.Tensor: Product of a and b as tensor of shape (..., 4).
    """
    aw, ax, ay, az = tf.unstack(a, axis=-1)
    bw, bx, by, bz = tf.unstack(b, axis=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return tf.stack([ow, ox, oy, oz], axis=-1)


def quaternion_multiply(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Multiply two quaternions representing rotations, returning the
    quaternion representing their composition, i.e. the versor with nonnegative
    real part.

    Example:

    .. code-block:: python

        a = tf.constant((1.,2.,3.,4.))
        b = tf.constant((5.,6.,7.,8.))
        quaternion_multiply(a=a, b=b)
        # <tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 60., -12., -30., -24.], dtype=float32)>

    Args:
        a (tf.Tensor): First quaternion with real part first, as tensor
            of shape (..., 4).
        b (tf.Tensor): Second quaternion with real part first, as tensor
            of shape (..., 4).

    Returns:
        tf.Tensor: Product of a and b as tensor of shape (..., 4).
    """
    return standardize_quaternion(quaternion_raw_multiply(a, b))


def quaternion_invert(quaternion: tf.Tensor) -> tf.Tensor:
    """Given a quaternion representing rotation, get the quaternion
    representing its inverse.

    Example:

    .. code-block:: python

        quaternion = tf.constant((1.,2.,3.,4.))
        quaternion_invert(quaternion=quaternion)
        # <tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 1., -2., -3., -4.], dtype=float32)>

    Args:
        quaternion (tf.Tensor): Quaternions as tensor of shape (..., 4),
            with real part first, which must be versors (unit
            quaternions).

    Returns:
        tf.Tensor: The inverse, a tensor of quaternions of shape (...,
        4).
    """
    scaling = tf.cast(tf.constant([1, -1, -1, -1]), dtype=quaternion.dtype)
    return quaternion * scaling


def quaternion_apply(quaternion: tf.Tensor, point: tf.Tensor) -> tf.Tensor:
    """Apply the rotation given by a quaternion to a 3D point.

    Example:

    .. code-block:: python

        quaternion = tf.constant((1.,1.,1.,4.))
        point = tf.constant((1.,1.,1.))
        quaternion_apply(quaternion=quaternion, point=point)
        # <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-11.,   1.,  31.], dtype=float32)>

    Args:
        quaternion (tf.Tensor): Quaternions as tensor of shape (..., 4),
            with real part first
        point (tf.Tensor): Points as tensor of shape (..., 3)

    Returns:
        tf.Tensor: Tensor of rotated points of shape (..., 3).

    Raises:
        ValueError: If the last dimension of point is not 3.
    """
    if point.shape[-1] != 3:
        raise ValueError("Points must be 3D")

    real_parts = tf.zeros(tf.shape(point)[:-1] + (1,))
    point_as_quaternion = tf.concat([real_parts, point], axis=-1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: tf.Tensor) -> tf.Tensor:
    """Convert rotations given as axis/angle to quaternions.

        Example:

    .. code-block:: python

        axis_angle = tf.constant((1.,1.,1.))
        axis_angle_to_quaternion(axis_angle=axis_angle)
        # <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.64785933, 0.43980235, 0.43980235, 0.43980235], dtype=float32)>

    Args:
        axis_angle (tf.Tensor): Rotations given as a vector in axis
            angle form, as a tensor of shape (..., 3), where the
            magnitude is the angle turned anticlockwise in radians
            around the vector's direction.

    Returns:
        tf.Tensor: Quaternions as tensor of shape (..., 4), with real
        part first.
    """

    angles = tf.norm(axis_angle, ord=2, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    sin_half_angles_over_angles = tf.cast(tf.zeros_like(angles), dtype=tf.float32)
    sin_half_angles_over_angles = tf.math.sin(half_angles) / angles
    quaternions = tf.concat(
        [tf.math.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle: tf.Tensor) -> tf.Tensor:
    """Convert rotations given as axis/angle to rotation matrices.

    Example:

    .. code-block:: python

        axis_angle = tf.constant((1.,1.,1.))
        axis_angle_to_matrix(axis_angle=axis_angle)
        # <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
        # array([[ 0.22629571, -0.18300788,  0.9567122 ],
        #        [ 0.9567122 ,  0.22629571, -0.18300788],
        #        [-0.18300788,  0.9567122 ,  0.22629571]], dtype=float32)>

    Args:
        axis_angle (tf.Tensor): Rotations given as a vector in axis
            angle form, as a tensor of shape (..., 3), where the
            magnitude is the angle turned anticlockwise in radians
            around the vector's direction.

    Returns:
        tf.Tensor: Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def quaternion_to_axis_angle(quaternions: tf.Tensor) -> tf.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Example:

    .. code-block:: python

        quaternions = tf.constant((1.,1.,1.,4.))
        quaternion_to_axis_angle(quaternions=quaternions)
        # <tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 2.752039,  2.752039, 11.008156], dtype=float32)>

    Args:
        quaternions: Quaternions as tensor of shape (..., 4), with real
            part first.
        quaternion (tf.Tensor)

    Returns:
        tf.Tensor: Rotations given as a vector in axis angle form, as a
        tensor of shape (..., 3), where the magnitude is the angle
        turned anticlockwise in radians around the vector's direction.
    """

    norms = tf.norm(quaternions[..., 1:], ord=2, axis=-1, keepdims=True)
    half_angles = tf.math.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    sin_half_angles_over_angles = tf.cast(tf.zeros_like(angles), dtype=tf.float32)
    sin_half_angles_over_angles = tf.math.sin(half_angles) / angles
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: tf.Tensor) -> tf.Tensor:
    """Convert rotations given as rotation matrices to axis/angle.

    Example:

    .. code-block:: python

        matrix = tf.constant(
            [
                [
                    [0.15885946, -0.56794965, -0.48926896],
                    [-1.0064808, -0.39120296, 1.6047943],
                    [0.05503756, 0.817741, 0.4543775],
                ]
            ]
        )
        matrix_to_axis_angle(matrix)
        # <tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[-0.5801526,  3.2366152,  2.2535346]], dtype=float32)>

    :param: matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Args:
        matrix (tf.Tensor)

    Returns:
        tf.Tensor: Rotations given as a vector in axis angle form, as a
        tensor of shape (..., 3), where the magnitude is the angle
        turned anticlockwise in radians around the vector's direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: tf.Tensor) -> tf.Tensor:
    """Converts 6D rotation representation by Zhou et al. [1] to rotation
    matrix using Gram--Schmidt orthogonalization per Section B of [1].

    Example:

    .. code-block:: python

        d6 = tf.constant((1.,1.,1.,1.,1.,1.))
        rotation_6d_to_matrix(d6)
        # <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
        # array([[0.57735026, 0.57735026, 0.57735026],
        #        [0.57735026, 0.57735026, 0.57735026],
        #        [0.        , 0.        , 0.        ]], dtype=float32)>

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. On the Continuity of
    Rotation Representations in Neural Networks. IEEE Conference on Computer
    Vision and Pattern Recognition, 2019. Retrieved from http://arxiv.org/abs/1812.07035

    Args:
        d6 (tf.Tensor): 6D rotation representation as tensor of shape
            (..., 6).

    Returns:
        tf.Tensor: Rotation matrices as tensor of shape (..., 3, 3).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = tf.nn.l2_normalize(a1, axis=-1)
    b2 = a2 - tf.reduce_sum(b1 * a2, keepdims=True) * b1
    b2 = tf.linalg.normalize(b2)[0]
    b3 = tf.linalg.cross(b1, b2)
    return tf.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: tf.Tensor) -> tf.Tensor:
    """Converts rotation matrices to 6D rotation representation by Zhou et al.

    [1] by dropping the last row.

    Example:

    .. code-block:: python

        matrix = tf.constant([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        matrix_to_rotation_6d(matrix)
        # <tf.Tensor: shape=(6,), dtype=float32, numpy=array([2., 1., 1., 1., 2., 1.], dtype=float32)>

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. On the Continuity of
    Rotation Representations in Neural Networks. IEEE Conference on Computer
    Vision and Pattern Recognition, 2019. Retrieved from http://arxiv.org/abs/1812.07035

    Args:
        matrix (tf.Tensor): Rotation matrices as tensor of shape (...,
            3, 3).

    Returns:
        tf.Tensor: 6D rotation representation as tensor of shape (...,
        6).
    """
    batch_dim = matrix.shape[:-2]
    return tf.reshape(tf.identity(matrix[..., :2, :]), batch_dim + (6,))
