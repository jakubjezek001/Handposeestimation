from typing import Dict

import torch
from pytorch_lightning.core.lightning import LightningModule
from src.constants import HRNET_CONFIG
from src.models.base_model import BaseModel
from src.models.HRnet.pose_hrnet import get_pose_net
from src.models.spatial_2d_soft_argmax import spatial_soft_argmax2d
from src.models.utils import cal_l1_loss
from src.utils import read_yaml
from torch import Tensor


class HeatmapPoseModel(BaseModel):
    def __init__(self, config):
        LightningModule.__init__(self)
        self.config = config
        self.epsilon = 1e-6
        hrnet_config = read_yaml(HRNET_CONFIG)
        self.encoder = get_pose_net(hrnet_config.MODEL36, True)

    def forward(self, x: Tensor) -> Tensor:
        image_h, image_w = x.size()[-2:]
        x = self.encoder(x)
        h_star_2d, h_star_z = x[:, :21], x[:, 21:]
        out = self.heatmap_to_joints(h_star_2d, h_star_z, image_h, image_w)
        # print(out[0,0].detach()-out[0,1].detach())
        return out

    def heatmap_to_joints(
        self, h_star_2d: Tensor, h_star_z: Tensor, image_h: int, image_w: int
    ) -> Tensor:
        heat_h, heat_w = h_star_2d.size()[-2:]
        scale = (
            torch.tensor([image_h * 1.0 / heat_h, image_w * 1.0 / heat_w])
            .view(1, 1, 2)
            .to(h_star_2d.device)
        )
        h_2d = self.normalize_heatmap(h_star_2d)
        joints2d = spatial_soft_argmax2d(h_2d, normalized_coordinates=False)
        joints2d = joints2d * scale
        z_r = torch.sum(h_2d * h_star_z, dim=[2, 3]).view(-1, 21, 1)

        return torch.cat([joints2d, z_r], dim=2)

    def normalize_heatmap(self, heatmap: Tensor) -> Tensor:
        batch, channels, height, width = heatmap.size()
        heatmap = heatmap.view(batch, channels, -1)
        heatmap = torch.exp(heatmap - torch.max(heatmap, dim=-1, keepdim=True)[0])
        heatmap = heatmap / (heatmap.sum(dim=-1, keepdim=True) + self.epsilon)
        return heatmap.view(batch, channels, height, width)

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
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
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
            "loss_z_unscaled": loss_z_unscaled.detach(),
        }

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
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
