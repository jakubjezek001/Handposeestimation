from typing import Tuple
from matplotlib.transforms import TransformNode

import torch
from src.types import CAMERA_PARAM, JOINTS_3D, JOINTS_25D, SCALE
from src.data_loader.joints import Joints
from PIL import Image
from math import pi, sin, cos
from torchvision import transforms

JOINTS = Joints()
PARENT_JOINT = JOINTS.mapping.ait.wrist
CHILD_JOINT = JOINTS.mapping.ait.index_mcp


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
    # print(joints_25D)
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


def error_in_conversion(true_joints_3D: JOINTS_3D, cal_joints_3D: JOINTS_3D) -> float:
    """Calculates absolutes error between original 3D coordinates and
     the ones recovered from 2.5 Dimensions.

    Args:
        true_joints_3D (JOINTS_3D): Original 3D coordinares from the data, unscaled
        cal_joints_3D (JOINTS_3D): Calculated 3D coordinares from the 2.5D coordinates, unscaled

    Returns:
        float: Maximum percentage error between the conversion and the original.
    """
    error = torch.abs(cal_joints_3D - true_joints_3D)
    # error = torch.sum((cal_joints_3D - true_joints_3D)**2, 0)**0.5
    return torch.max(error)


def get_rotation_matrix(angle) -> torch.Tensor:
    """Retursn 2D rotation matrix

    Args:
        angle (int): Angle in degrees. Measured counterclockwise from the x axis.

    Returns:
        torch.Tensor: A 2x2 Rotation tensor.
    """
    deg = pi / 180
    return torch.tensor(
        [[cos(angle * deg), -sin(angle * deg)], [sin(angle * deg), cos(angle * deg)]]
    )


def sample_rotator(
    image: Image.Image, joints: JOINTS_25D, angle: int
) -> Tuple[Image.Image, JOINTS_25D]:
    """Rotates the sample image and the 2D keypoints by 'angle' in degrees counter clockwise to x axis around
    the image center. The relative depth is not changed.


    Args:
        image (Image.Image): a PIL image, preferable uncropped
        joints (JOINTS_25D): Tensor of all 2.5 D coordinates.
        angle (int): Angle in degrees.

    Returns:
        Tuple[Image.Image, JOINTS_25D]: Rotated image and keypoints.
    """
    rot_mat = get_rotation_matrix(angle)
    joints_rotated = joints.clone()
    # centering joints at image center.
    joints_rotated[:, :-1] = joints_rotated[:, :-1] - image.size[0] / 2
    joints_rotated[:, :-1] = (rot_mat @ joints_rotated[:, :-1].T).T
    # reverting back to original origin i,e. top left corner.
    joints_rotated[:, :-1] = joints_rotated[:, :-1] + image.size[0] / 2
    # Rotate image by the same angle, make sure expand is set to False.
    # Also angle here is measured in clockwise direction make sure to add a minus sign.
    image_rotated = transforms.functional.rotate(image, -angle, expand=False)
    return image_rotated, joints_rotated


def sample_cropper(
    image: Image.Image,
    joints: JOINTS_25D,
    crop_margin: float = 1.5,
    crop_joints: bool = True,
) -> Tuple[Image.Image, JOINTS_25D]:
    top, left = torch.min(joints[:, 1]), torch.min(joints[:, 0])
    bottom, right = torch.max(joints[:, 1]), torch.max(joints[:, 0])
    height, width = bottom - top, right - left
    origin_x = int(left - width * (crop_margin - 1) / 2)
    origin_y = int(top - height * (crop_margin - 1) / 2)
    joints_cropped = joints.clone()
    img_crop = transforms.functional.crop(
        image,
        top=origin_y,
        left=origin_x,
        height=int(height * crop_margin),
        width=int(width * crop_margin),
    )
    if crop_joints:
        joints_cropped[:, 0] = joints_cropped[:, 0] - origin_x
        joints_cropped[:, 1] = joints_cropped[:, 1] - origin_y
    return img_crop, joints_cropped


def sample_resizer(
    image: Image.Image,
    joints: JOINTS_25D,
    shape: Tuple = (128, 128),
    resize_joints: bool = True,
) -> Tuple[Image.Image, JOINTS_25D]:
    """Resizes the sample to given size.

    Args:
        image (Image.Image): A PIL image
        joints (JOINTS_25D): 2.5D joints. The depth is kept as is.
        shape (Tuple, optional): Size to which the image should be reshaped. Defaults to (128, 128).
        resize_joints (bool, optional): To resize the joints along with the image.. Defaults to True.

    Returns:
        Tuple[Image.Image, JOINTS_25D]: REsized image and keypoints.
    """
    width, height = image.size
    image = transforms.functional.resize(image, shape)
    joints_resized = joints.clone()
    if resize_joints:
        joints_resized[:, 0] = joints_resized[:, 0] * 128 / (width)
        joints_resized[:, 1] = joints_resized[:, 1] * 128 / (height)
    return image, joints_resized
