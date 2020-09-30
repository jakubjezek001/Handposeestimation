import torch
from torch.nn import L1Loss


def cal_l1_loss(pred_joints: torch.Tensor, true_joints: torch.Tensor, alpha: float):
    """Calculates L1 loss between the predicted and true joints.  The relative depth (Z)
    is penalized seperately.

    Args:
        pred_joints (torch.Tensor): Predicted 2.5D joints.
        true_joints (torch.Tensor): True 2.5D joints.
        alpha (float): Coefficient of loss term for depth

    Returns:
        [type]: [description]
    """
    pred_uv = pred_joints[:, :, :-1]
    pred_z = pred_joints[:, :, -1:]
    true_uv = true_joints[:, :, :-1]
    true_z = true_joints[:, :, -1:]
    loss = L1Loss()
    return loss(pred_uv, true_uv) + alpha * loss(pred_z, true_z)
