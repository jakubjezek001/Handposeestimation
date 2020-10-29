from typing import List
import math
import torch
import torchvision
from easydict import EasyDict as edict
from pytorch_lightning.core.lightning import LightningModule
from src.models.utils import cal_l1_loss
from src.utils import get_console_logger
from torch import nn
from torch.nn import functional as F
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


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
        self.console_logger = get_console_logger("baseline_model")
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        if not self.config["resnet_trainable"]:
            self.console_logger.warning("Freeizing the underlying  Resnet !")
            for param in self.resnet18.parameters():
                param.requires_grad = False
        self.resnet18.fc = nn.Sequential()
        self.final_layers = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 21 * 3)
        )
        self.train_metrics_epoch = None
        self.train_metrics = None
        self.validation_metrics_epoch = None
        self.plot_params = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, width, height = x.size()
        x = self.resnet18(x)
        x = self.final_layers(x)
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
            dict: Output dictionary containng the calculated metrics. Must have a key
                "loss".
                This is the key that is optimized
        """
        x, y, scale = batch["image"], batch["joints"], batch["scale"]
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(prediction, y, scale)
        loss = loss_2d + self.config.alpha * loss_z
        self.train_metrics = {
            "loss": loss.detach(),
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
            "loss_z_unscaled": loss_z_unscaled.detach(),
        }
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}
        return {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
        }

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_z = torch.stack([x["loss_z"] for x in outputs]).mean()
        loss_2d = torch.stack([x["loss_2d"] for x in outputs]).mean()
        loss_z_unscaled = torch.stack([x["loss_z_unscaled"] for x in outputs]).mean()
        self.train_metrics_epoch = {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
        }

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=["bias", "bn"]
    ):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.config.batch_size
        self.train_iters_per_epoch = self.config.num_samples // global_batch_size

    def configure_optimizers(self):
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.config.opt_weight_decay
        )
        optimizer = LARSWrapper(
            torch.optim.Adam(
                parameters, lr=self.config.lr * math.sqrt(self.config.batch_size)
            )
        )
        warmup_epochs = self.config.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0,
        )

        scheduler = {
            "scheduler": linear_warmup_cosine_decay,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """This method is called at every validation step (i.e. every batch in an
            validation epoch).
            Only metrics and loss are calulated which are later averaged by the
            validation_epoch_end().
        Args:
            batch (dict): Batch of the validation samples. Must have "image" and
                "joints".
            batch_idx (int): The index of batch in an epoch.

        Returns:
            dict: Output dictionary containng the calculated metrics.
        """
        x, y, scale = batch["image"], batch["joints"], batch["scale"]
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(prediction, y, scale)
        loss = loss_2d + self.config.alpha * loss_z
        metrics = {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
        }
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}

        return metrics

    def validation_epoch_end(self, outputs: List[dict]) -> dict:
        """This function called at the end of the validation epoch, i.e. passing through
            all the
        validation batches in an epoch

        Args:
            outputs (List[dict]): validation_step() output from all batches of an epoch.

        Returns:
            dict: Dictionary containing all the parameters from validation_step()
                averaged.
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_z = torch.stack([x["loss_z"] for x in outputs]).mean()
        loss_2d = torch.stack([x["loss_2d"] for x in outputs]).mean()
        loss_z_unscaled = torch.stack([x["loss_z_unscaled"] for x in outputs]).mean()
        metrics = {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
        }
        self.validation_metrics_epoch = metrics
