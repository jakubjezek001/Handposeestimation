from typing import Tuple

import torch
from src.constants import CHILD_JOINT, PARENT_JOINT
from src.types import CAMERA_PARAM, JOINTS_3D, JOINTS_25D, SCALE


def convert_to_2_5D(K: CAMERA_PARAM, joints_3D: JOINTS_3D) -> Tuple[JOINTS_25D, SCALE]:
    """Converts coordinates from 3D to 2.5D
    Refer: https://arxiv.org/pdf/1804.09534.pdf

    Args:
        K (CAMERA_PARAM):3x3 Matrix with camera parameters.
        joints_3D (JOINTS_3D): Original 3D coordinates unscaled.

    Returns:
        Tuple[JOINTS_25D, SCALE]: 2.5 D coordinates and scale information.
    """
    scale = (((joints_3D[CHILD_JOINT] - joints_3D[PARENT_JOINT]) ** 2).sum()) ** 0.5
    joints_25D = ((K @ (joints_3D.T)).T) / joints_3D[:, -1:]
    joints_25D[:, -1] = (joints_3D[:, -1] - joints_3D[0, -1]) / scale
    return joints_25D, scale


def convert_2_5D_to_3D(
    joints_25D: JOINTS_25D, scale: SCALE, K: CAMERA_PARAM
) -> JOINTS_3D:
    """Converts coordinates from 2.5 Dimesnions to original 3 Dimensions.
    Refer: https://arxiv.org/pdf/1804.09534.pdf

    Args:
        joints_25D (JOINTS_25D): 2.5 D coordinates.
        scale (SCALE): Eucledian distance between the parent and child joint.
        K (CAMERA_PARAM): 3x3 Matrix with camera parameters.

    Returns:
        JOINTS_3D: Obtained 3D coordinates from 2.5D coordinates and scale information.
    """
    Z_root, K_inv = get_root_depth(joints_25D, K)
    Z_coord = (joints_25D[:, -1:] + Z_root) * scale
    camera_projection = joints_25D.clone()
    camera_projection[:, -1] = 1
    joints_3D = ((K_inv @ (camera_projection.T)).T) * Z_coord
    return joints_3D


def get_root_depth(
    joints_25D: JOINTS_25D, K: CAMERA_PARAM
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the scale normalized  Z_root from the joints coordinates using the result
    in https://arxiv.org/pdf/1804.09534.pdf equation 6 and 7.
    Note: There is a correction that needs to be made in the paper. x_n, y_n, x_m and y_m
    are the camera projections multiplued with inverted camera parameters.

    Args:
        joints_25D (JOINTS_25D): 21 joint coordinates in 2.5 Dimensions.
        K (CAMERA_PARAM): [description] : camera parameters of the data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: scaled root Z coordinate and inverted camera parameters.
    """
    K_inv = torch.inverse(K)
    x_n, y_n, _ = K_inv @ torch.cat(
        (joints_25D[PARENT_JOINT, :-1], torch.tensor([1.0])), 0
    )
    Z_n = joints_25D[PARENT_JOINT, -1]
    x_m, y_m, _ = K_inv @ torch.cat(
        (joints_25D[CHILD_JOINT, :-1], torch.tensor([1.0])), 0
    )
    Z_m = joints_25D[CHILD_JOINT, -1]
    C = 1
    a = (x_n - x_m) ** 2 + (y_n - y_m) ** 2
    b = Z_n * (x_n ** 2 + y_n ** 2 - x_n * x_m - y_n * y_m) + Z_m * (
        x_m ** 2 + y_m ** 2 - x_n * x_m - y_n * y_m
    )
    c = (
        (x_n * Z_n - x_m * Z_m) ** 2
        + (y_n * Z_n - y_m * Z_m) ** 2
        + (Z_n - Z_m) ** 2
        - C
    )
    # print("a={},b={},c={}".format(a, b, c))
    # print("x_n={}, y_n={}, Z_n={}".format(x_n, y_n, Z_n))
    # print("x_m={}, y_m={}, Z_m={}".format(x_m, y_m, Z_m))
    Z_root = 0.5 * (-b + (b ** 2 - 4 * a * c) ** 0.5) / a
    return Z_root, K_inv


def error_in_conversion(true_joints_3D: JOINTS_3D, K: CAMERA_PARAM) -> float:
    """Calculates maximum percentage error between original 3D coordinates and
     the ones recovered from 2.5 Dimensions.

    Args:
        true_joints_3D (JOINTS_3D): Original 3D coordinares from the data, unscaled
        K (CAMERA_PARAM): camera parameters for those coordinates.

    Returns:
        float: Maximum percentage error between the conversion and the original.
    """
    joints25D, scale = convert_to_2_5D(K, true_joints_3D)
    error_percentage = (
        torch.abs((convert_2_5D_to_3D(joints25D, scale, K) - true_joints_3D))
        / true_joints_3D
    )
    return torch.max(error_percentage) * 100
