import torch
import torchvision
from easydict import EasyDict as edict
from src.models.base_model import BaseModel
from src.models.utils import cal_l1_loss
from src.utils import get_console_logger
from torch import nn
from typing import Dict


class BaselineModel(BaseModel):
    def __init__(self, config: edict):
        """Class wrapper for the fully supervised model used as baseline.
        It uses Resnet as the base model.
        Appends more layers in the end to fit the HPE Task.
        Args:
            config (dict): Model configurations passed as an easy dict.
        """
        super().__init__(config)
        self.console_logger = get_console_logger("baseline_model")
        self.encoder = torchvision.models.resnet18(pretrained=False)
        if not self.config["resnet_trainable"]:
            self.console_logger.warning("Freeizing the underlying  Resnet!")
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.encoder.fc = nn.Sequential()
        self.final_layers = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 21 * 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.final_layers(x)
        x = x.view(-1, 21, 3)
        return x

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
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

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, torch.Tensor]:
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
