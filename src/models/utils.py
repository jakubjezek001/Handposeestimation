from src.visualization.visualize import plot_truth_vs_prediction
import torch
from torch.nn import L1Loss
from comet_ml import Experiment


def cal_l1_loss(pred_joints: torch.Tensor, true_joints: torch.Tensor):
    """Calculates L1 loss between the predicted and true joints.  The relative depth (Z)
    is penalized seperately.

    Args:
        pred_joints (torch.Tensor): Predicted 2.5D joints.
        true_joints (torch.Tensor): True 2.5D joints.

    Returns:
        [type]: [description]
    """
    pred_uv = pred_joints[:, :, :-1]
    pred_z = pred_joints[:, :, -1:]
    true_uv = true_joints[:, :, :-1]
    true_z = true_joints[:, :, -1:]
    loss = L1Loss()
    return loss(pred_uv, true_uv), loss(pred_z, true_z)


def calculate_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, step: str = "train"
) -> dict:
    """Calculates the metrics on a batch of predicted and true labels.

    Args:
        y_pred (torch.Tensor): Batch of predicted labels.
        y_true (torch.Tensor): Batch of True labesl.
        step (str, optional): This argument specifies whether the metrics are caclulated for
            train or val set. Appends suitable name to the keys in returned dictionary.
            Defaults to "train".

    Returns:
        dict: Calculated metrics as a dictionary.
    """
    distance_joints = (
        torch.sum(((y_pred - y_true) ** 2), 2) ** 0.5
    )  # shape: (batch, 21)
    mean_distance = torch.mean(distance_joints)
    median_distance = torch.median(distance_joints)
    return {f"EPE_mean_{step}": mean_distance, f"EPE_median_{step}": median_distance}


def log_metrics(metrics: dict, comet_logger: Experiment, epoch: int, context_val: bool):
    if context_val:
        with comet_logger.validate():
            comet_logger.log_metrics(metrics, epoch=epoch)
    else:
        with comet_logger.train():
            comet_logger.log_metrics(metrics, epoch=epoch)


def log_image(prediction, y, x, gpu: bool, context_val: bool, comet_logger: Experiment):
    if gpu:
        pred_label = prediction.data[0].cpu().numpy()
        true_label = y.data[0].cpu().detach().numpy()
    else:
        pred_label = prediction[0].detach().numpy()
        true_label = y[0].detach().numpy()
    if context_val:
        with comet_logger.validate():
            plot_truth_vs_prediction(
                pred_label, true_label, x.data[0].cpu(), comet_logger
            )
    else:
        with comet_logger.train():
            plot_truth_vs_prediction(
                pred_label, true_label, x.data[0].cpu(), comet_logger
            )
