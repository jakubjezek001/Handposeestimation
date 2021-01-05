import math

import torch
import torchvision
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from src.models.utils import (
    vanila_contrastive_loss,
    translate_encodings,
    rotate_encoding,
)
from torch import nn
from torch.nn import functional as F
import cv2


class Hybrid2Model(LightningModule):
    """
    Hybrid version for simCLRr implementation inspired from paper
    https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    The equivariance is preserved by transforming projection space.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder()
        self.projection_head = self.get_projection_head()
        # Variables used by callbacks
        self.train_metrics_epoch = None
        self.train_metrics = None
        self.validation_metrics_epoch = None
        self.plot_params = None

    def get_encoder(self):
        encoder = torchvision.models.resnet18(pretrained=True)
        # Removing the last classification layer.
        encoder.fc = nn.Sequential()
        return encoder

    def get_projection_head(self):
        projection_head = nn.Sequential(
            nn.Linear(
                self.config.projection_head_input_dim,
                self.config.projection_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.projection_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.projection_head_hidden_dim,
                self.config.output_dim,
                bias=False,
            ),
        )
        return projection_head

    def get_transformed_projections(self, batch):
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
            # jitter_x = torch.cat((batch["jitter_x_1"], batch["jitter_x_2"]), dim=0)
            # jitter_y = torch.cat((batch["jitter_y_1"], batch["jitter_y_2"]), dim=0)

            max_projections = torch.max(projections, dim=1).values

            jitter_x = (
                torch.cat(
                    (
                        batch["jitter_x_1"] / float(image1_shape[0]),
                        batch["jitter_x_2"] / float(image2_shape[0]),
                    ),
                    dim=0,
                )
                * max_projections[:, 0]
            )
            jitter_y = (
                torch.cat(
                    (
                        batch["jitter_y_1"] / float(image1_shape[1]),
                        batch["jitter_y_2"] / float(image2_shape[1]),
                    ),
                    dim=0,
                )
                * max_projections[:, 1]
            )
            # moving the encodings by same amount.
            projections = translate_encodings(projections, -jitter_x, -jitter_y)

        projections = projections.view((batch_size * 2, -1))
        projection1 = F.normalize(projections[:batch_size])
        projection2 = F.normalize(projections[batch_size:])
        return projection1, projection2

    def contrastive_step(self, batch):
        projection1, projection2 = self.get_transformed_projections(batch)
        loss = vanila_contrastive_loss(projection1, projection2)
        return loss

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return {"embedding": self.encoder(x), "projection": projection}

    def training_step(self, batch, batch_idx):
        loss = self.contrastive_step(batch)
        self.train_metrics = {"loss": loss.detach()}
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "params": {k: v for k, v in batch.items() if "image" not in k},
        }
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.train_metrics_epoch = {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.contrastive_step(batch)
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "params": {k: v for k, v in batch.items() if "image" not in k},
        }
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("checkpoint_saving_loss", loss)
        self.validation_metrics_epoch = {"loss": loss}

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
        # excuding bias and batch norm from the weight decay.
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.config.opt_weight_decay
        )
        # Applying LARS to all other layers.
        # lr = 0.075* sqrt(batch_size) Appendix B of the paper.
        # optimizer = LARSWrapper(
        #     torch.optim.Adam(parameters, lr=0.075 * math.sqrt(self.config.batch_size))
        # )
        optimizer = LARSWrapper(
            torch.optim.Adam(
                parameters,
                lr=self.config.lr
                * math.sqrt(self.config.batch_size * self.config.num_of_mini_batch),
            )
        )

        # The schdeuler is called after every step in an epoch hence adjusting the
        # warmup epochs param.
        warmup_epochs = (
            self.config.warmup_epochs
            * self.train_iters_per_epoch
            // self.config.num_of_mini_batch
        )
        max_epochs = (
            self.trainer.max_epochs
            * self.train_iters_per_epoch
            // self.config.num_of_mini_batch
        )

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
            "frequency": 1.0,
        }
        return [optimizer], [scheduler]
