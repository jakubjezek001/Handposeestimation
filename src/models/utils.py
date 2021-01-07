import os
from typing import Tuple
import torch
import numpy as np
from comet_ml import Experiment
from src.constants import SAVED_MODELS_BASE_PATH
from src.visualization.visualize import (
    plot_simclr_images,
    plot_truth_vs_prediction,
    plot_pairwise_images,
    plot_hybrid2_images,
)
from torch.nn import L1Loss


def cal_l1_loss(
    pred_joints: torch.Tensor, true_joints: torch.Tensor, scale: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates L1 loss between the predicted and true joints.  The relative unscaled
    depth (Z) is penalized seperately.

    Args:
        pred_joints (torch.Tensor): Predicted 2.5D joints.
        true_joints (torch.Tensor): True 2.5D joints.
        scale (torch.Tensor): Scale to unscale the z coordinate. If not provide unscaled
            loss_z is returned, otherwise scaled loss_z is returned.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 2d loss, scaled z relative
            loss and unscaled z relative loss.
    """
    if scale is None:
        scale = 1.0
    pred_uv = pred_joints[:, :, :-1]
    pred_z = pred_joints[:, :, -1:]
    true_uv = true_joints[:, :, :-1]
    true_z = true_joints[:, :, -1:]
    loss = L1Loss()
    return (
        loss(pred_uv, true_uv),
        loss(pred_z, true_z),
        loss(pred_z * scale, true_z * scale),
    )


def calculate_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, step: str = "train"
) -> dict:
    """Calculates the metrics on a batch of predicted and true labels.

    Args:
        y_pred (torch.Tensor): Batch of predicted labels.
        y_true (torch.Tensor): Batch of True labesl.
        step (str, optional): This argument specifies whether the metrics are caclulated
            for train or val set. Appends suitable name to the keys in returned
            dictionary. Defaults to "train".

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


def log_simclr_images(img1, img2, context_val: bool, comet_logger: Experiment):

    if context_val:
        with comet_logger.validate():
            plot_simclr_images(img1.data[0].cpu(), img2.data[0].cpu(), comet_logger)
    else:
        with comet_logger.train():
            plot_simclr_images(img1.data[0].cpu(), img2.data[0].cpu(), comet_logger)


def vanila_contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature=0.5
) -> torch.Tensor:
    """Calculates the contrastive loss as mentioned in SimCLR paper
        https://arxiv.org/pdf/2002.05709.pdf.
    Parts of the code adapted from pl_bolts nt_ext_loss.

    Args:
        z1 (torch.Tensor): Tensor of normalized projections of the images.
            (#samples_in_batch x vector_dim).
        z2 (torch.Tensor): Tensor of normalized projections of the same images but with
            different transformation.(#samples_in_batch x vector_dim)
        temperature (float, optional): Temperature term in the contrastive loss.
            Defaults to 0.5. In SimCLr paper it was shown t=0.5 is good for training
            with small batches.

    Returns:
        torch.Tensor: Contrastive loss (1 x 1)
    """
    z = torch.cat([z1, z2], dim=0)
    n_samples = len(z)

    # Full similarity matrix
    cov = torch.mm(z, z.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()
    return loss


def get_latest_checkpoint(experiment_name: str, checkpoint: str = "") -> str:
    """Path to the last saved checkpoint of the trained model.

    Args:
        experiment_name (str): experiment name.

    Returns:
        str: absolute path to the latest checkpoint
    """
    checkpoint_path = os.path.join(
        SAVED_MODELS_BASE_PATH, experiment_name, "checkpoints"
    )
    if checkpoint == "":
        latest_checkpoint = sorted(os.listdir(checkpoint_path))[-1]
    else:
        latest_checkpoint = checkpoint
    return os.path.join(checkpoint_path, latest_checkpoint)


def log_pairwise_images(img1, img2, gt_pred, context_val, comet_logger):
    gt_pred = {
        k: [v[0].data[0].cpu().numpy(), v[1].data[0].cpu().numpy()]
        for k, v in gt_pred.items()
    }
    if context_val:
        with comet_logger.validate():
            plot_pairwise_images(
                img1.data[0].cpu(), img2.data[0].cpu(), gt_pred, comet_logger
            )
    else:
        with comet_logger.train():
            plot_pairwise_images(
                img1.data[0].cpu(), img2.data[0].cpu(), gt_pred, comet_logger
            )


def log_hybrid2_images(img1, img2, params, context_val, comet_logger):
    params = {k: v.data[0].cpu() for k, v in params.items()}
    if context_val:
        with comet_logger.validate():
            plot_hybrid2_images(
                img1.data[0].cpu(), img2.data[0].cpu(), params, comet_logger
            )
    else:
        with comet_logger.train():
            plot_hybrid2_images(
                img1.data[0].cpu(), img2.data[0].cpu(), params, comet_logger
            )


def get_rotation_2D_matrix(
    angle: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Generates 2D rotation matrix transpose. the matrix generated is for the whole batch.
    The implementation of 2D matrix is same as that in openCV.

    Args:
        angle (torch.Tensor): 1D tensor of rotation angles for the batch
        center_x (torch.Tensor): 1D tensor of x coord of center of the keypoints.
        center_y (torch.Tensor): 1D tensor of x coord of center of the keypoints.
        scale (torch.Tensor): Scale, set it to 1.0.

    Returns:
        torch.Tensor: Returns a tensor of 2D rotation matrix for the batch.
    """
    # convert to radians
    angle = angle * np.pi / 180
    alpha = scale * torch.cos(angle)
    beta = scale * torch.sin(angle)
    rot_mat = torch.zeros((len(angle), 3, 2))
    rot_mat[:, :, 0] = torch.stack(
        [alpha, beta, (1 - alpha) * center_x - beta * center_y], dim=1
    )
    rot_mat[:, :, 1] = torch.stack(
        [-beta, alpha, (1 - alpha) * center_y + beta * center_x], dim=1
    )

    return rot_mat


def rotate_encoding(encoding, angle) -> torch.Tensor:
    """Function to 2D rotate a batch of encodings by a batch of angles.
    The third dimension is n not changed.

    Args:
        encoding (torch.Tensor): Encodings of shape (batch_size,m,3)
        angle ([type]): batch of angles (batch_size,)

    Returns:
        torch.Tensor: Rotated batch of keypoints.
    """
    # center_xyz = torch.mean(encoding.detach(), 1)
    # rot_mat = get_rotation_2D_matrix(
    #     angle, center_xyz[:, 0], center_xyz[:, 1], scale=1.0
    # )
    # rot_mat = rot_mat.cuda(encoding.device) if encoding.is_cuda else rot_mat
    # encoding[:, :, :2] = torch.bmm(
    #     torch.cat((encoding[:, :, :2], torch.ones_like(encoding[:, :, -1:])), dim=2),
    #     rot_mat,
    # )
    # return encoding
    center_xyz = torch.mean(encoding, 1)
    rot_mat = get_rotation_2D_matrix(
        angle, center_xyz[:, 0], center_xyz[:, 1], scale=1.0
    )
    encoding_z = encoding[:, :, -1:].clone()
    encoding[:, :, -1] = 1.0
    rot_mat = rot_mat.cuda(encoding.device) if encoding.is_cuda else rot_mat
    encoding_xy = torch.bmm(encoding, rot_mat)

    return torch.cat([encoding_xy, encoding_z], dim=-1)


def translate_encodings(
    encoding: torch.Tensor, translate_x: torch.Tensor, translate_y: torch.Tensor
) -> torch.Tensor:
    """Translates the encodings along first two dimensions with linear scaling

    Args:
        encoding (torch.Tensor): image encodings/projections from the network
        translate_x (torch.Tensor): normlaized jitter along x axis of the input image
        translate_y (torch.Tensor): normalized jitter along y axis of the input image.

    Returns:
        torch.Tensor: Translated encodings based on scaled normalized jitter.
    """
    max_encodings = torch.max(encoding, dim=1).values
    min_encodings = torch.min(encoding, dim=1).values
    encoding[:, :, 0] += (
        translate_x * (max_encodings[:, 0] - min_encodings[:, 0])
    ).view((-1, 1))
    encoding[:, :, 1] += (
        translate_y * (max_encodings[:, 1] - min_encodings[:, 1])
    ).view((-1, 1))
    return encoding
