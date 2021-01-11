from typing import Dict, Tuple

import torch
from easydict import EasyDict as edict
from src.models.simclr_model import SimCLR
from src.models.utils import (
    rotate_encoding,
    translate_encodings,
    vanila_contrastive_loss,
)
from torch import Tensor
from torch.nn import functional as F


class Hybrid2Model(SimCLR):
    """
    Hybrid version for simCLRr implementation inspired from paper
    https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    The equivariance is preserved by transforming projection space.
    """

    def __init__(self, config: edict):
        super().__init__(config)

    def get_transformed_projections(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_transform = torch.cat(
            (batch["transformed_image1"], batch["transformed_image2"]), dim=0
        )
        # Assuming shape of batch is [batch, channels, width, height]
        image1_shape = batch["transformed_image1"].size()[-2:]
        image2_shape = batch["transformed_image2"].size()[-2:]
        batch_size = int(len(batch_transform) / 2)
        encodings = self.encoder(batch_transform)
        projections = self.projection_head(encodings).view((batch_size * 2, -1, 3))

        if "rotate" in self.config.augmentation:
            # make sure the shape is (batch,-1,3).
            angles = torch.cat((batch["angle_1"], batch["angle_2"]), dim=0)
            # rotating the projections in opposite direction
            projections = rotate_encoding(projections, -angles)
        if "crop" in self.config.augmentation:
            # normalizing jitter with respect to image size.
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
            # moving the encodings by same amount.
            projections = translate_encodings(projections, -jitter_x, -jitter_y)

        projections = projections.view((batch_size * 2, -1))
        projection1 = F.normalize(projections[:batch_size])
        projection2 = F.normalize(projections[batch_size:])
        return projection1, projection2

    def contrastive_step(self, batch: Dict[str, Tensor]) -> Tensor:
        projection1, projection2 = self.get_transformed_projections(batch)
        loss = vanila_contrastive_loss(projection1, projection2)
        return loss
