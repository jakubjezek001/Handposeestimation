import json
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from comet_ml import Experiment
from src.constants import MASTER_THESIS_DIR
from src.data_loader.joints import Joints
from src.types import JOINTS_3D, JOINTS_25D
from src.utils import read_json
from torchvision import transforms

joints = Joints()


def plot_hand(
    axis: plt.Axes, coords_hand: np.array, plot_3d: bool = False, linewidth: str = "1"
):
    """Makes a hand stick figure from the coordinates wither in uv plane or xyz plane on the passed axes object.
    Code adapted from:  https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py

    Args:
        axis (plt.Axes): Matlplotlib axes, for 3D plots pass axes with 3D projection
        coords_hand (np.array): 21 coordinates of hand as numpy array. (21 x 3). Expects AIT format.
        plot_3d (bool, optional): Pass this as true for using the the depth parameter to plot the hand. Defaults to False.
        linewidth (str, optional): Linewidth to be used for drawing connecting bones. Defaults to "1".
    """

    colors = np.array(
        read_json(
            os.path.join(MASTER_THESIS_DIR, "src", "visualization", "joint_color.json")
        )["joint_colors"]
    )
    coords_hand = joints.ait_to_freihand(coords_hand)
    # define connections and colors of the bones
    bones = [
        ((i, i + 1), colors[1 + i, :]) if i % 4 != 0 else ((0, i + 1), colors[1 + i, :])
        for i in range(0, 20)
    ]
    # Making connection between the joints.
    for connection, color in bones:
        coord1 = coords_hand[connection[0], :]
        coord2 = coords_hand[connection[1], :]
        coords = np.stack([coord1, coord2])
        if plot_3d:
            axis.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                color=color,
                linewidth=linewidth,
            )
        else:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)

    # Highlighting the joints
    for i in range(21):
        if plot_3d:
            axis.plot(
                coords_hand[i, 0],
                coords_hand[i, 1],
                coords_hand[i, 2],
                "o",
                color=colors[i, :],
            )
        else:
            axis.plot(coords_hand[i, 1], coords_hand[i, 0], "o", color=colors[i, :])


def plot_truth_vs_prediction(
    y_pred: Union[JOINTS_25D, JOINTS_3D],
    y_true: Union[JOINTS_25D, JOINTS_3D],
    image: torch.Tensor,
    experiment: Experiment,
):
    """Generates the graphics with input image, predicetd labels and the ground truth.

    Args:
        y_pred (Union[JOINTS_25D, JOINTS_3D]): Output from the model as a tensor. shape (21 x 3)
        y_true (Union[JOINTS_25D, JOINTS_3D]): ground truth. shape(21 x 3)
        image (torch.Tensor): Input image to the model.
        experiment (Experiment): Comet ml experiment object.
    """
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(131)
    plt.imshow(transforms.ToPILImage()(image))
    ax0.title.set_text("Input image")
    ax1 = fig.add_subplot(132)
    plot_hand(ax1, y_true)
    ax1.title.set_text("True joints")
    ax2 = fig.add_subplot(133)
    plot_hand(ax2, y_pred)
    ax2.title.set_text("Predicted joints")
    experiment.log_figure(figure=plt)
