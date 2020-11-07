import os
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from comet_ml import Experiment
from pytorch_lightning.loggers import comet
from src.constants import MASTER_THESIS_DIR
from src.data_loader.joints import Joints
from src.types import JOINTS_3D, JOINTS_25D
from src.utils import read_json
from torchvision import transforms

joints = Joints()


def plot_hand(
    axis: plt.Axes,
    coords_hand: np.array,
    plot_3d: bool = False,
    linewidth: str = "1",
    linestyle: str = "-",
    alpha: float = 1.0,
):
    """Makes a hand stick figure from the coordinates wither in uv plane or xyz plane on the passed axes object.
    Code adapted from:  https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py

    Args:
        axis (plt.Axes): Matlplotlib axes, for 3D plots pass axes with 3D projection
        coords_hand (np.array): 21 coordinates of hand as numpy array. (21 x 3). Expects AIT format.
        plot_3d (bool, optional): Pass this as true for using the the depth parameter to plot the hand. Defaults to False.
        linewidth (str, optional): Linewidth to be used for drawing connecting bones. Defaults to "1".
        linestyle (str, optional): MAtplotlib linestyle, Defaults to ":"
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
                linestyle=linestyle,
                alpha=alpha,
            )
        else:
            axis.plot(
                coords[:, 0],
                coords[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )

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
            axis.plot(
                coords_hand[i, 0],
                coords_hand[i, 1],
                "o",
                color=colors[i, :],
                linestyle=linestyle,
                alpha=alpha,
            )


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
    img = cv2.cvtColor(np.array(transforms.ToPILImage()(image)), cv2.COLOR_BGR2RGB)
    width, height, _ = img.shape
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    plt.imshow(img)
    plot_hand(ax1, y_true)
    ax1.title.set_text("True joints")
    ax2 = fig.add_subplot(122)
    plot_hand(ax2, y_true, alpha=0.2, linestyle=":")
    plot_hand(ax2, y_pred)
    ax2.set_xlim([0, width])
    ax2.set_ylim([height, 0])
    ax2.title.set_text("Predicted joints")
    if experiment is not None:
        experiment.log_figure(figure=plt)
    plt.close()


def plot_simclr_images(img1: np.array, img2: np.array, comet_logger: Experiment):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121)
    plt.imshow(
        cv2.cvtColor(np.array(transforms.ToPILImage()(img1.cpu())), cv2.COLOR_BGR2RGB)
    )
    ax.set_title("Image 1")
    ax = fig.add_subplot(122)
    plt.imshow(
        cv2.cvtColor(np.array(transforms.ToPILImage()(img2.cpu())), cv2.COLOR_BGR2RGB)
    )
    ax.set_title("Image 2")
    if comet_logger is not None:
        comet_logger.log_figure(figure=plt)
    plt.close()


def plot_pairwise_images(img1, img2, gt_pred, comet_logger):

    fig = plt.figure(figsize=(10, 10))
    title = "\n".join([f"{k}: gt:{v[0]}, pred:{v[1]}" for k, v in gt_pred.items()])
    ax = fig.add_subplot(121)
    plt.imshow(
        cv2.cvtColor(np.array(transforms.ToPILImage()(img1.cpu())), cv2.COLOR_BGR2RGB)
    )
    ax.set_title("Image 1")
    ax = fig.add_subplot(122)
    plt.imshow(
        cv2.cvtColor(np.array(transforms.ToPILImage()(img2.cpu())), cv2.COLOR_BGR2RGB)
    )
    ax.set_title("Image 2")
    fig.suptitle(title)
    if comet_logger is not None:
        comet_logger.log_figure(figure=plt)

    plt.close()
