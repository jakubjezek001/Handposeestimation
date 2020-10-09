import torch
from comet_ml import Experiment
from src.visualization.visualize import plot_truth_vs_prediction
from torch.nn import L1Loss


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


def vanila_contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature=0.5
) -> torch.Tensor:
    """Calculates the contrastive loss as mentioned in SimCLR paper
        https://arxiv.org/pdf/2002.05709.pdf.
    Parts of the code adapted from pl_bolts nt_ext_loss.

    Args:
        z1 (torch.Tensor): Tensor of projections of the images. (#samples_in_batch x vector_dim).
        z2 (torch.Tensor): Tensor of projections of the same images but with different transformation.
             (#samples_in_batch x vector_dim)
        temperature (float, optional): Temperature term in the contrastive loss. Defaults to 0.5.

    Returns:
        torch.Tensor: Contrastive loss (1 x 1)
    """

    # Normalizing the vectors.
    z1 = z1 / torch.norm(z1, dim=1).view((-1, 1))
    z2 = z2 / torch.norm(z2, dim=1).view((-1, 1))

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
