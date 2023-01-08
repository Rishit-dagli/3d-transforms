import tensorflow as tf


def quaternion_to_matrix(quaternions: tf.Tensor) -> tf.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Example:

    .. code-block:: python

        quaternion = tf.constant([0.0, 0.0, 0.0, 4.0])
        output = tensorflow3d_transforms.quaternion_to_matrix(quaternions=quaternion)
        # <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
        # array([[-1.,  0.,  0.],
        #     [ 0., -1.,  0.],
        #     [ 0.,  0.,  1.]], dtype=float32)>


    :param quaternions: A tensor of shape (..., 4) representing quaternions with real part first.
    :type quaternions: tf.Tensor
    :return: A tensor of shape (..., 3, 3) representing rotation matrices.
    :rtype: tf.Tensor
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
    """
    Convert rotations given as rotation matrices to quaternions.

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


    :param matrix: A tensor of shape (..., 3, 3) representing rotation matrices.
    :type matrix: tf.Tensor
    :return: A tensor of shape (..., 4) representing quaternions with real part first.
    :rtype: tf.Tensor
    :raises ValueError: If the shape of the input matrix is invalid that is does not have the shape (..., 3, 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = tf.unstack(tf.reshape(matrix, batch_dim + (9,)), axis=-1)

    q_abs = _sqrt_positive_part(tf.stack(
    [
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ],
    axis=-1,
    ))

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
    quat_candidates = quat_by_rijk / (2.0 * tf.reduce_max(q_abs[..., None], flr, keepdims=True))
    max_indices = tf.argmax(q_abs, axis=-1)
    one_hot = tf.one_hot(max_indices, depth=4)
    selected = tf.boolean_mask(quat_candidates, one_hot > 0.5)
    return tf.reshape(selected, batch_dim + [4])

def _sqrt_positive_part(x: tf.Tensor) -> tf.Tensor:
    """
    Returns the square root of all positive elements of x and 0 for others.

    :param x: A tensor
    :type x: tf.Tensor
    :return: A tensor with the same shape as x
    :rtype: tf.Tensor
    """
    ret = tf.zeros_like(x)
    positive_mask = x > 0
    ret = tf.where(positive_mask, tf.math.sqrt(x), ret)
    return ret

def _axis_angle_rotation(axis: str, angle: tf.Tensor) -> tf.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    :param axis: The axis about which the rotation is performed. Must be one of 'X', 'Y', 'Z'.
    :type axis: str
    :param angle: Any shape tensor of Euler angles in radians
    :type angle: tf.Tensor
    :return: A tensor of shape (..., 3, 3) representing rotation matrices.
    :rtype: tf.Tensor
    :raises ValueError: If the axis is not one of 'X', 'Y', 'Z'.
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
    """
    Convert rotations given as euler angles to rotation matrices.

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

        euler_angles_to_matrix(euler_angles, convention="XYZ")
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
    
    :param euler_angles: A tensor of shape (..., 3) representing euler angles.
    :type euler_angles: tf.Tensor
    :param convention: The euler angle convention. A string containing a combination of three uppercase letters from {"X", "Y", and "Z"}.
    :type convention: str
    :return: A tensor of shape (..., 3, 3) representing rotation matrices.
    :rtype: tf.Tensor
    :raises ValueError: If the shape of the input euler angles is invalid that is does not have the shape (..., 3).
    :raises ValueError: If the convention string is invalid that is does not have the length 3.
    :raises ValueError: If the second character of the convention string is the same as the first or third.
    :raises ValueError: If the convention string contains characters other than {"X", "Y", and "Z"}.
    """

    if euler_angles.shape[-1] != 3:
        raise ValueError(f"Invalid euler angle shape {euler_angles.shape}, last dimension should be 3.")
    if len(convention) != 3:
        raise ValueError(f"Invalid euler angle convention {convention}, should be a string of length 3.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid euler angle convention {convention}, second character should be different from first and third.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, tf.unstack(euler_angles, axis=-1))
    ]

    return tf.linalg.matmul(tf.linalg.matmul(matrices[0], matrices[1]), matrices[2])