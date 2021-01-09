from typing import Dict

import torch
from easydict import EasyDict as edict
from src.data_loader.utils import PARENT_JOINT, get_root_depth
from src.models.baseline_model import BaselineModel
from src.models.utils import cal_l1_loss
from src.utils import get_console_logger
from torch import nn
from torch.nn.modules.loss import L1Loss


class DenoisedBaselineModel(BaselineModel):
    def __init__(self, config: edict):
        """Class wrapper for the baseline supervised model.
        Uses Resnet as the base model.
        Appends more layers in the end to fit the HPE Task.
        Args:
            config (dict): Model configurations passed as an easy dict.
        """
        super().__init__(config)
        self.console_logger = get_console_logger("denoised_baseline_model")
        self.denoiser = nn.Sequential(
            nn.Linear(21 * 3 + 3 * 3 + 1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        x, y, scale, k = batch["image"], batch["joints"], batch["scale"], batch["K"]
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(prediction, y, scale)
        loss = loss_2d + self.config.alpha * loss_z

        z_root_denoised = self.get_denoised_z_root_calc(prediction.detach(), k)

        z_root_gt = batch["joints3D"][:, PARENT_JOINT, -1] / scale
        loss_z_denoise = L1Loss()(z_root_gt, z_root_denoised.view(-1))
        loss += loss_z_denoise

        self.train_metrics = {
            "loss": loss.detach(),
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
            "loss_z_unscaled": loss_z_unscaled.detach(),
            "loss_z_denoise": loss_z_denoise.detach(),
        }
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}
        return {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
            "loss_z_denoise": loss_z_denoise,
        }

    def get_denoised_z_root_calc(
        self, joints25D: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:

        z_root_calc, k_inv = get_root_depth(joints25D, k, is_batch=True)
        z_root_calc = z_root_calc.view((-1, 1))
        batch_size = joints25D.size()[0]
        denoising_input = torch.cat(
            (
                z_root_calc,
                k_inv.reshape((batch_size, -1)),
                joints25D.reshape(batch_size, -1),
            ),
            dim=1,
        )
        return self.denoiser(denoising_input.detach()) + z_root_calc.detach()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y, scale, k = batch["image"], batch["joints"], batch["scale"], batch["K"]
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(prediction, y, scale)
        loss = loss_2d + self.config.alpha * loss_z

        z_root_denoised = self.get_denoised_z_root_calc(y, k)

        z_root_gt = batch["joints3D"][:, PARENT_JOINT, -1] / scale
        loss_z_denoise = L1Loss()(z_root_gt, z_root_denoised.view(-1))
        loss += loss_z_denoise
        metrics = {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
            "loss_z_denoise": loss_z_denoise,
        }
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}

        return metrics
