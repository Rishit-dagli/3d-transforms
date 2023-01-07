import tensorflow as tf


def quaternion_to_matrix(quaternions: tf.Tensor) -> tf.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

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
