from typing import Tuple, Union

import numpy as np
import torch
from pytorch_lightning.core.lightning import LightningModule
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import convert_2_5D_to_3D
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_epe_statistics(
    predictions: torch.tensor, ground_truth: torch.tensor, dim: int
) -> dict:
    """Calculates the eucledian diatnce statistics between the all coordinates. In case of 2.5 D

    Args:
        predictions (torch.tensor): Predicted coordinates  of shape (#sample x 21 x 3)
        ground_truth (torch.tensor): True coordinates of shape (#samples x 21 x3)
        dim (int): to denote if the predictions and ground truth are 2.5D or 3D.
            If 2 is passed .

    Returns:
        dict: Returns a dictionary containing following keys
                'mean_epe', 'median_epe', 'min_epe', 'max_epe'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dim == 2:
        predictions_ = predictions[:, :, :-1].clone()
        ground_truth_ = ground_truth[:, :, :-1].clone()
    else:
        if dim != 3:
            print("Coordinates treated as 3D")
        predictions_ = predictions.clone()
        ground_truth_ = ground_truth.clone()

    with torch.no_grad():
        eucledian_dist = (
            torch.sum(((predictions_.to(device) - ground_truth_.to(device)) ** 2), 2)
            ** 0.5
        )
        mean_epe = torch.mean(eucledian_dist)
        median_epe = torch.median(eucledian_dist)
        max_epe = torch.max(eucledian_dist)
        min_epe = torch.min(eucledian_dist)

    return {
        "eucledian_dist": eucledian_dist,
        "mean": mean_epe,
        "median": median_epe,
        "min": min_epe,
        "max": max_epe,
    }


def calculate_predicted_3D(
    joints: torch.tensor, camera_params: torch.tensor, scales: torch.tensor
) -> torch.tensor:
    """calculates the 3D joints from 2.5D joints.

    Args:
        joints (torch.tensor): predicted joints in 2.5D (#sample x 21 x 3)
        camera_params (torch.tensor): camera prameters (#sample x 3 x 3)
        scales (torch.tensor): scale for the jointss (#sample x 1)

    Returns:
        torch.tensor: predicted joints in 3D (#sample x 21 x 3)
    """
    predicted_3D_coords = []
    for i in tqdm(range(len(joints))):
        predicted_3D_coords.append(
            convert_2_5D_to_3D(
                joints[i].to(torch.device("cpu")),
                scales[i].to(torch.device("cpu")),
                camera_params[i].to(torch.device("cpu")),
            )
        )
    return torch.stack(predicted_3D_coords, axis=0)


def get_predictions_and_ground_truth(
    model: LightningModule, data: Data_Set, **dataloader_args
) -> dict:
    """calculates the predictions by providing the model input image. Also prepares
    the necessary transformations required for calculating the statistucs.

    Args:
        model (LightningModule): A model defined using pytorch lightening.
        data (Data_Set): the data for which the model should be evaluated.

    **dataloader_args: Argumenst for torch.utls.data.DataLoader.
        Adjust num_workers and batch_size for speed.

    Returns:
        dict: dict with lists of predictions and ground truth. Following keys are
        present.
        "predictions","ground_truth","ground_truth_3d",
        "ground_truth_recreated_3d","predictions_3d","camera_param" and "scale"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    data_loader = DataLoader(data, **dataloader_args)
    # initilaize the lists
    predictions = []
    ground_truth = []
    ground_truth_3d = []
    ground_truth_recreated_3d = []
    scale = []
    camera_param = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader)):
            input_tensor = batch["image"].to(device)
            ground_truth.append(batch["joints"])
            ground_truth_3d.append(batch["joints3D"])
            ground_truth_recreated_3d.append(batch["joints3D_recreated"])
            scale.append(batch["scale"])
            camera_param.append(batch["K"])
            predictions.append(model(input_tensor))
    predictions = torch.cat(predictions, axis=0)
    scale = torch.cat(scale, axis=0)
    camera_param = torch.cat(camera_param, axis=0)
    predictions_3d = calculate_predicted_3D(predictions, camera_param, scale)
    ground_truth = torch.cat(ground_truth, axis=0)
    ground_truth_3d = torch.cat(ground_truth_3d, axis=0)
    ground_truth_recreated_3d = torch.cat(ground_truth_recreated_3d, axis=0)
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "ground_truth_3d": ground_truth_3d,
        "ground_truth_recreated_3d": ground_truth_recreated_3d,
        "predictions_3d": predictions_3d,
        "camera_param": camera_param,
        "scale": scale,
    }


def evaluate(model: LightningModule, data: Data_Set, **dataloader_args) -> dict:
    """Computes the predictions and various statistics.

    Args:
        model (LightningModule): Trained model.
        data (Data_Set): data set for evaluation.

    Returns:
        dict: dictionary containing evaluation
    """
    prediction_dict = get_predictions_and_ground_truth(model, data, **dataloader_args)
    epe_2D = calculate_epe_statistics(
        prediction_dict["predictions"], prediction_dict["ground_truth"], dim=2
    )
    epe_3D = calculate_epe_statistics(
        prediction_dict["predictions_3d"], prediction_dict["ground_truth_3d"], dim=3
    )
    epe_3D_recreated = calculate_epe_statistics(
        prediction_dict["predictions_3d"],
        prediction_dict["ground_truth_recreated_3d"],
        dim=3,
    )
    epe_3D__gt_vs_3D_recreated = calculate_epe_statistics(
        prediction_dict["ground_truth_3d"],
        prediction_dict["ground_truth_recreated_3d"],
        dim=3,
    )
    return {
        "Mean_EPE_2D": epe_2D["mean"].cpu(),
        "Median_EPE_2D": epe_2D["median"].cpu(),
        "Mean_EPE_3D": epe_3D["mean"].cpu(),
        "Median_EPE_3D": epe_3D["median"].cpu(),
        "Mean_EPE_3D_R": epe_3D_recreated["mean"].cpu(),
        "Median_EPE_3D_R": epe_3D_recreated["median"].cpu(),
        "Mean_EPE_3D_R_v_3D": epe_3D__gt_vs_3D_recreated["mean"].cpu(),
        "Median_EPE_3D_R_V_3D": epe_3D__gt_vs_3D_recreated["median"].cpu(),
    }


def get_pck_curves(
    eucledian_dist: torch.Tensor,
    threshold_min: float = 0.0,
    threshold_max: float = 0.5,
    step: float = 0.005,
    per_joint: bool = False,
) -> Tuple[np.array, np.array]:
    """Calculates pck curve i.e. percentage of predicted keypoints under a certain
    threshold of eucledian distance from the ground truth. The number of thresholds this
    is calculated depends upon threshold_max, threshold_min and step.

    Args:
        eucledian_dist (torch.Tensor): Eucldeian distance between ground truth and
            predictions. (#samples x 21 x 3)
        threshold_min (float, optional):Minumum threshold that should be tested.
            Defaults to 0.0.
        threshold_max (float, optional):Maximum threshold to be tested. Defaults to 0.5.
        step (float, optional): Defaults to 0.005.
        per_joint (bool, optional):If true calculates it seperately for 21 joints.
            Defaults to False.

    Returns:
        Tuple[np.array, np.array]: Returns pck curve (#num_of_thresholds) or
            (21 x #num_of_thresholds) and corresponding thesholds (#num_of_thresholds).
    """
    thresholds = np.arange(threshold_min, threshold_max, step)
    if per_joint:
        percent_under_threshold = np.array(
            [
                torch.mean((eucledian_dist < theta) * 1.0, axis=0).cpu().numpy().T
                for theta in thresholds
            ]
        ).T
    else:
        percent_under_threshold = np.array(
            [
                torch.mean((eucledian_dist < theta) * 1.0).cpu().numpy()
                for theta in thresholds
            ]
        )
    return percent_under_threshold, thresholds


def cal_auc_joints(
    eucledian_dist: torch.Tensor, per_joint=True
) -> Union[np.array, float]:
    """Calculates Area Under the Curve (AUC) for pck curve of the eucledian distance between
    predictions and ground truth.

    Args:
        eucledian_dist (torch.Tensor): Eucldeian distance between ground truth and
            predictions. (#samples x 21 x 3)
        per_joint (bool, optional):If true calculates it seperately for 21 joints.
            Defaults to True.

    Returns:
        Union[np.array, float]: Either return AUC per joint or overall AUC.
    """
    percent_index_threshold, thresholds = get_pck_curves(
        eucledian_dist, threshold_min=0.0, threshold_max=0.5, step=0.005, per_joint=True
    )
    normalizing_factor = np.trapz(y=np.ones(len(thresholds)), x=thresholds)
    auc_per_joint = np.array(
        [
            np.trapz(y=percent_index_threshold[i], x=thresholds) / normalizing_factor
            for i in range(21)
        ]
    )
    if per_joint:
        return auc_per_joint
    else:
        return np.mean(auc_per_joint)
