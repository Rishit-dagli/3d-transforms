import tensorflow as tf

from .so3_ops import _so3_exp_map, hat, so3_log_map


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V
