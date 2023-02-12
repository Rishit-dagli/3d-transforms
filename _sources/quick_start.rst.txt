Quickstart
==========

Install TensorFlow3d Transforms:

.. code-block:: bash

    pip install tensorflow3dtransforms

You can now try out any of the method examples in the documentation. Here are a few starter example you could try out:

1. Define a set of random rotations and use the exponential map to map the parameters to matrices in SO(3). After which we use the logarithmic map to map the matrices back to parameters to identify any relations. We now briefly show what happens in this example:

A rotation matrix :math:`R` in 3D space can be parameterized using logarithmic map or exponential map of a 3D vector :math:`\theta`. Exponential map of a 3D vector :math:`\theta` is calculated as:

.. math::

    R = e^{\theta} = I + \frac{\sin(\left|\theta\right|)}{\left|\theta\right|}\theta + \frac{1-\cos(\left|\theta\right|)}{\left|\theta\right|^2}\theta^2

where :math:`I` is the identity matrix.

We then compute the logarithmic map of a rotation matrix :math:`R` which is the inverse operation of exponential map. The logarithmic map of :math:`R` is calculated as:

.. math::

    \theta = \text{logmap}(R) = \frac{A}{\sin(\left|A\right|)}\text{tan}^{-1}\left(\frac{\text{trace}(R)-1}{2}\right)

where :math:`A = \left[R_{32}-R_{23}, R_{13}-R_{31}, R_{21}-R_{12}\right]^T``

The code examples below shows this in action with the TensorFlow3d Transforms library on 10,000 points.

.. code-block:: python

    import tensorflow3dtransforms as t3d
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    thetas = tf.random.uniform(shape=(10000, 3), minval=-np.pi, maxval=np.pi)
    Rs = t3d.so3_exp_map(thetas)
    thetas_recovered = t3d.so3_log_map(Rs)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(thetas[:, 0], thetas[:, 1], thetas[:, 2], c="blue")
    ax[1].scatter(
        thetas_recovered[:, 0], thetas_recovered[:, 1], thetas_recovered[:, 2], c="red"
    )
    ax[0].set_title("Original parameters")
    ax[1].set_title("Recovered parameters")
    plt.show()

.. image:: _static/so3_log_map.png

2. We calculate a 3D rotation matrix :math:`R` is calculated using the ``axis_angle_to_matrix`` function from the ``tensorflow3dtransforms`` library. A set of points points are then transformed by this rotation matrix to obtain their rotated versions.

The conversion from axis-angle representation to rotation matrix is done using:

.. math::

    R = \begin{bmatrix} \cos(\theta) + u_x^2(1-\cos(\theta)) & u_xu_y(1-\cos(\theta)) - u_z\sin(\theta) & u_xu_z(1-\cos(\theta)) + u_y\sin(\theta) \\ u_yu_x(1-\cos(\theta)) + u_z\sin(\theta) & \cos(\theta) + u_y^2(1-\cos(\theta)) & u_yu_z(1-\cos(\theta)) - u_x\sin(\theta) \\ u_zu_x(1-\cos(\theta)) - u_y\sin(\theta) & u_zu_y(1-\cos(\theta)) + u_x\sin(\theta) & \cos(\theta) + u_z^2(1-\cos(\theta))
    \end{bmatrix}

where :math:`u_x`, :math:`u_y`, :math:`u_z` are the components of the unit vector axis and :math:`\theta` is the angle of rotation.

The code examples below shows this in action with the TensorFlow3d Transforms library.

.. code-block:: python

    import tensorflow3dtransforms as t3d
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    axis = np.array([0, 0, 1])  # Z-axis
    angle = np.pi / 4  # 45 degrees

    R = axis_angle_to_matrix(axis * angle)
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    rotated_points = np.dot(points, R)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c="b", marker="o", label="Original Points"
    )
    ax.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        rotated_points[:, 2],
        c="r",
        marker="x",
        label="Rotated Points",
    )

    # Add labels and show the plot
    plt.legend(loc="upper left")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

.. image:: _static/axis_angle_to_matrix.png

3. Here's an example of visualizing a 3D rotation map as a heatmap.

.. code-block:: python

    import tensorflow3dtransforms as t3d
    import numpy as np
    import matplotlib.pyplot as plt

    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    angle = np.pi / 4
    R = t3d.axis_angle_to_matrix(axis * angle)

    fig, ax = plt.subplots()
    im = ax.imshow(R, cmap='hot')
    fig.colorbar(im)
    plt.show()

.. image:: _static/axis_angle_to_matrix_heatmap.png

These were some examples just to get you started with using some of the APIs in this library, and there are a lot more possibilities to what you could do with the library.