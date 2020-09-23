import torch
from typing import NewType, Tuple

#  Type definition
JOINTS_25D = NewType(
    "Joints_25D", torch.tensor
)  # shape (points, 3) , third dimension set to unity.
SCALE = NewType("scale", torch.tensor)  # shape (1)
JOINTS_3D = NewType("Joints_3D", torch.tensor)  # shape  (pints, 3)


def convert_to_2_5D(
    camera_param: torch.tensor, joints_3D: torch.tensor
) -> Tuple[JOINTS_25D, SCALE]:
    # Using the distance between and second coordinate as scale. (i.e. length of index fingure.)
    scale = (((joints_3D[0] - joints_3D[2]) ** 2).sum()) ** 0.5
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
    depth = joints_25D[:, -1:] * scale  # Z = (Z/s)*s
    joints_25D[:, -1] = 1.0  # Making 2.5D -> 2D.
    joints_3D = (
        (torch.inverse(camera_param) @ (joints_25D.T)).T
    ) * depth  # J_3D = Z *(K^-1)*J_2D
    return joints_3D
