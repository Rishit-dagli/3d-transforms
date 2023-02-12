Quickstart
==========

1. Install TensorFlow3d Transforms:

.. code-block:: bash

    pip install tensorflow3dtransforms

2. Try out any of the method examples in the documentation. Here are a few starter example you could try out:

Define a set of random rotations and use the exponential map to map the parameters to matrices in SO(3). After which we use the logarithmic map to map the matrices back to parameters to identify any relations:

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

Here's an example of generating a set of points and rotating them.

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

Here's an example of visualizing a 3D rotation map as a heatmap.

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

These were some examples just to get you started with using some of the APIs in this library.