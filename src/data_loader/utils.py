import torch
from typing import NewType, Tuple

#  Type definition
JOINTS_25D = NewType(
    "Joints_25D", torch.tensor
)  # shape (points, 3) , third dimension set to unity.
SCALE = NewType("scale", torch.tensor)  # shape (1)
JOINTS_3D = NewType("Joints_3D", torch.tensor)  # shape  (pints, 3)
PARENT_JOINT = 0
CHILD_JOINT = 2


def convert_to_2_5D(
    camera_param: torch.tensor, joints_3D: torch.tensor
) -> Tuple[JOINTS_25D, SCALE]:
    # Using the distance between and second coordinate as scale. (i.e. length of index fingure.)
    scale = (((joints_3D[CHILD_JOINT] - joints_3D[PARENT_JOINT]) ** 2).sum()) ** 0.5
    # Using the equation (1/Z)*K@(J_3D) = J_2D
    # Below Joints3D is the tensor of 21 x 3, where as J_3D is 3 x 1.
    # joints3D.T -> 3 x 21
    # camera param and K -> 3x3
    #  camera_param@(joints3D.T).T -> 21 x 3
    # joints3D[:,-1:] -> 21 x 1
    # joints25D -> 21 x3 with the third dimesion as 1.
    joints_25D = ((camera_param @ (joints_3D.T)).T) / joints_3D[:, -1:]
    # adding the .5 th dimension relative to the root, here it is the 0th coordinate.
    # scale normalization of the relative z component. As the xy coordinate are not affected by scale.
    joints_25D[:, -1] = (joints_3D[:, -1] - joints_3D[0, -1]) / scale
    return joints_25D, scale


def convert_2_5D_to_3D(
    joints_25D: JOINTS_25D, scale: SCALE, camera_param: torch.tensor
) -> JOINTS_3D:
    # Assuming the depth coordinates are relative.
    # TODO: Incorrect, check it with Adrian.
    # depth = joints_25D[:, -1:] * scale  # Z = (Z/s)*s
    # joints_25D[:, -1] = 1.0  # Making 2.5D -> 2D.
    # joints_3D = (
    #     (torch.inverse(camera_param) @ (joints_25D.T)).T
    # ) * depth  # J_3D = Z *(K^-1)*J_2D
    Z_root = get_root_depth(joints_25D)
    Z_coord = (joints_25D[:, -1:] + Z_root) * scale
    camera_projection = joints_25D.clone()
    camera_projection[:, -1] = 1
    joints_3D = ((torch.inverse(camera_param) @ (camera_projection.T)).T) * Z_coord
    return joints_3D


def get_root_depth(joints_25D: JOINTS_25D) -> torch.Tensor:
    """Gets the scale normalized  Z_root from the joints coordinates using the result
    in https://arxiv.org/pdf/1804.09534.pdf equation 6 and 7

    Args:
        joints_25D (JOINTS_25D): 21 joint coordinates in 2.5 Dimensions.
            Where Z coordinate is relative to root and scale normalized (21 x 3)

    Returns:
        torch.Tensor:Scaled root Z coordinate. (1)
    """

    x_n, y_n, Z_n = joints_25D[CHILD_JOINT, :]
    x_m, y_m, Z_m = joints_25D[PARENT_JOINT, :]
    # print(x_n)
    # print(x_n, y_m, Z_m)
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
    Z_root = 0.5 * (-b + (b ** 2 - 4 * a * c) ** 0.5) / a
    # print(a,b,c)
    # print(Z_root)
    return Z_root
