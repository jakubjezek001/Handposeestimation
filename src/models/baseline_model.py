from typing import List

import torch
import torchvision
from easydict import EasyDict as edict
from pytorch_lightning.core.lightning import LightningModule
from src.visualization.visualize import plot_truth_vs_prediction
from torch.nn import functional as F


class BaselineModel(LightningModule):
    def __init__(self, config: edict):
        """Class wrapper for the baseline supervised model.
        Uses Resnet as the base model.
        Appends more layers in the end to fit the HPE Task.
        Args:
            config (dict): Model configurations passed as an easy dict.
        """
        super().__init__()
        self.config = config
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        if ~self.config["resnet_trainable"]:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 128)
        self.layer_1 = torch.nn.Linear(128, 128)
        self.output_layer = torch.nn.Linear(128, 21 * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, width, height = x.size()
        x = self.resnet18(x)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = x.view(batch_size, 21, 3)
        return x

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """This method is called at every training step (i.e. every batch in an epoch)
        It specifies the logs of trainig steps and the loss that needs to be optimized.
        Note: Graphics for first sample of few batches are logged, namely 1 and 4.

        Args:
            batch (dict): Batch of the training samples. Must have "image" and "joints".
            batch_idx (int): The index of batch in an epoch.

        Returns:
            dict: Output dictionary containng the calculated metrics. Must have a key "loss".
                This is the key that is optimized
        """
        x, y = batch["image"], batch["joints"]
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        train_metrics = self.calculate_metrics(prediction, y, step="train")
        comet_experiment = self.logger.experiment
        comet_experiment.log_metrics({**{"loss": loss}, **train_metrics})
        if batch_idx == 1 or batch_idx == 4:
            if self.config.gpu:
                pred_label = prediction.data[0].cpu().numpy()
                true_label = y.data[0].cpu().detach().numpy()
            else:
                pred_label = prediction[0].detach().numpy()
                true_label = y[0].detach().numpy()

            plot_truth_vs_prediction(
                pred_label, true_label, x.data[0].cpu(), comet_experiment
            )
        return {**{"loss": loss}, **train_metrics}

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """This method is called at every validation step (i.e. every batch in an validation epoch).
            Only metrics and loss are calulated which are later averaged by the validation_epoch_end().
        Args:
            batch (dict): Batch of the validation samples. Must have "image" and "joints".
            batch_idx (int): The index of batch in an epoch.

        Returns:
            dict: Output dictionary containng the calculated metrics.
        """
        x, y = batch["image"], batch["joints"]
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        val_metrics = self.calculate_metrics(prediction, y, step="val")
        comet_experiment = self.logger.experiment
        if batch_idx == 1 or batch_idx == 4:
            if self.config.gpu:
                pred_label = prediction.data[0].cpu().numpy()
                true_label = y.data[0].cpu().detach().numpy()
            else:
                pred_label = prediction[0].detach().numpy()
                true_label = y[0].detach().numpy()

            plot_truth_vs_prediction(
                pred_label, true_label, x.data[0].cpu(), comet_experiment
            )
        return {**{"val_loss": loss}, **val_metrics}

    def validation_epoch_end(self, outputs: List[dict]) -> dict:
        """This function called at the end of the validation epoch, i.e. passing through all the
        validation batches in an epoch

        Args:
            outputs (List[dict]): validation_step() output from all batches of an epoch.

        Returns:
            dict: Dictionary containing all the parameters from validation_step() averaged.
        """
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_epe_mean = torch.stack([x["EPE_mean_val"] for x in outputs]).mean()
        val_epe_median = torch.stack([x["EPE_median_val"] for x in outputs]).mean()
        self.logger.experiment.log_metrics(
            {
                "val_loss": val_loss,
                "val_epe_mean": val_epe_mean,
                "val_epe_median": val_epe_median,
            }
        )
        return {
            "val_loss": val_loss,
            "val_epe_mean": val_epe_mean,
            "val_epe_median": val_epe_median,
        }

    def calculate_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, step: str = "train"
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
        return {
            f"EPE_mean_{step}": mean_distance,
            f"EPE_median_{step}": median_distance,
        }
