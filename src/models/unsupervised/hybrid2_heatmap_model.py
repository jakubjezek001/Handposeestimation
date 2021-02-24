from argparse import HelpFormatter
import kornia
from src.models.utils import (
    get_heatmap_transformation_matrix,
    vanila_contrastive_loss,
    affine_mat_to_theta,
)
from typing import Dict, Tuple

from easydict import EasyDict as edict
from src.models.unsupervised.hybrid2_model import Hybrid2Model
from src.models.unsupervised.simclr_heatmap_model import SimCLRHeatmap
from torch import Tensor
import torch
from torch.nn import functional as F


class Hybrid2HeatmapModel(SimCLRHeatmap, Hybrid2Model):
    """
    Hybrid2 version for simCLRr implementation inspired from paper
    https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    The equivariance is preserved by transforming projection space.
    """

    def __init__(self, config: edict):
        SimCLRHeatmap.__init__(self, config)

    def contrastive_step(self, batch: Dict[str, Tensor]) -> Tensor:
        if self.config.preserve_heatmap:
            heatmap1, heatmap2, mask = self.get_transformed_heatmaps(batch)
            if self.config.use_mask:
                heatmap1 = heatmap1 * mask
                heatmap2 = heatmap2 * mask
            heatmap1 = F.normalize(heatmap1)
            heatmap2 = F.normalize(heatmap2)
            loss = vanila_contrastive_loss(heatmap1, heatmap2)
            return loss
        else:
            return Hybrid2Model.contrastive_step(self, batch)

    def get_transformed_heatmaps(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_transform = torch.cat(
            (batch["transformed_image1"], batch["transformed_image2"]), dim=0
        )
        batch_size = len(batch_transform) // 2
        encoder_heatmaps = self.encoder(batch_transform)
        projected_heatmaps = self.projection_head(encoder_heatmaps)

        heatmap_dim = torch.tensor(projected_heatmaps.size()[-2:])
        # prepare transform param
        jitter_x, jitter_y, scale, angle = self.prepare_transformation_param(batch)
        # Applying transformations
        # NOTE : All the params are in opposite direction because torch uses opposite direction for affine_grid.
        transform_mat = get_heatmap_transformation_matrix(
            jitter_x=jitter_x,
            jitter_y=jitter_y,
            scale=1 / scale,
            angle=angle,
            heatmap_dim=heatmap_dim.to(angle.device),
        )
        grid = F.affine_grid(
            affine_mat_to_theta(transform_mat, heatmap_dim[0], heatmap_dim[1]),
            size=projected_heatmaps.shape,
        )
        projected_heatmaps = F.grid_sample(projected_heatmaps, grid.detach())
        featuremap1 = projected_heatmaps[:batch_size].view(
            -1, heatmap_dim[0] * heatmap_dim[1]
        )
        featuremap2 = projected_heatmaps[batch_size:].view(
            -1, heatmap_dim[0] * heatmap_dim[1]
        )
        mask = F.grid_sample(torch.ones_like(projected_heatmaps), grid)
        mask = (mask[:batch_size] * mask[batch_size:]).view(
            -1, heatmap_dim[0] * heatmap_dim[1]
        )
        return featuremap1, featuremap2, mask

    def prepare_transformation_param(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns jitter, scale and angle parameter.
        Supplements them with default values if nothing provided.

        Args:
            batch (): batch from dataloader

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: jitter_x,jitter_y,scale,angle
        """
        # Assuming shape of batch is [batch, channels, height, width]
        image1_shape = batch["transformed_image1"].size()[-2:]
        image2_shape = batch["transformed_image2"].size()[-2:]
        batch_size = len(batch["transformed_image1"])
        scale = torch.cat(
            (batch["crop_margin_scale_1"], batch["crop_margin_scale_2"]), dim=0
        )
        if "crop" in self.config.augmentation:
            jitter_x = torch.cat(
                (
                    batch["jitter_x_1"] / float(image1_shape[0]),
                    batch["jitter_x_2"] / float(image2_shape[0]),
                ),
                dim=0,
            )
            jitter_y = torch.cat(
                (
                    batch["jitter_y_1"] / float(image1_shape[1]),
                    batch["jitter_y_2"] / float(image2_shape[1]),
                ),
                dim=0,
            )
        else:
            jitter_x = torch.zeros(batch_size * 2).float()
            jitter_y = torch.zeros(batch_size * 2).float()
        if "rotate" in self.config.augmentation:
            angle = torch.cat((batch["angle_1"], batch["angle_2"]), dim=0)
        else:
            angle = torch.zeros(batch_size * 2).float()
        return jitter_x.view(-1, 1), jitter_y.view(-1, 1), scale.view(-1, 1), angle
