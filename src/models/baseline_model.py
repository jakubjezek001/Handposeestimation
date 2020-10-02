from typing import List

import torch
import torchvision
from easydict import EasyDict as edict
from pytorch_lightning.core.lightning import LightningModule
from src.visualization.visualize import plot_truth_vs_prediction
from torch.nn import functional as F
from src.models.utils import cal_l1_loss, log_metrics, log_image


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
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        if ~self.config["resnet_trainable"]:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 128)
        self.layer_1 = torch.nn.Linear(128, 128)
        self.output_layer = torch.nn.Linear(128, 21 * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, width, height = x.size()
        x = self.resnet18(x)
        x = F.relu(x)
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
        loss_2d, loss_z = cal_l1_loss(prediction, y)
        loss = loss_2d + self.config.alpha * loss_z
        context_val = False
        comet_logger = self.logger.experiment
        metrics = {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}
        log_metrics(metrics, comet_logger, self.current_epoch, context_val)
        if batch_idx == 1 or batch_idx == 4:
            log_image(prediction, y, x, self.config.gpu, context_val, comet_logger)
        return metrics

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
        loss_2d, loss_z = cal_l1_loss(prediction, y)
        loss = loss_2d + self.config.alpha * loss_z
        context_val = False
        comet_logger = self.logger.experiment
        metrics = {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}
        if batch_idx == 1 or batch_idx == 4:
            log_image(prediction, y, x, self.config.gpu, context_val, comet_logger)
        return metrics

    def validation_epoch_end(self, outputs: List[dict]) -> dict:
        """This function called at the end of the validation epoch, i.e. passing through all the
        validation batches in an epoch

        Args:
            outputs (List[dict]): validation_step() output from all batches of an epoch.

        Returns:
            dict: Dictionary containing all the parameters from validation_step() averaged.
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_z = torch.stack([x["loss_z"] for x in outputs]).mean()
        loss_2d = torch.stack([x["loss_2d"] for x in outputs]).mean()
        metrics = {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}
        comet_logger = self.logger.experiment
        context_val = True
        log_metrics(metrics, comet_logger, self.current_epoch, context_val)
        return metrics
