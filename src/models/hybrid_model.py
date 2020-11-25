import math

import torch
import torchvision
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import L1Loss, CrossEntropyLoss
from src.models.utils import vanila_contrastive_loss


class HybridModel(LightningModule):
    """
    Hybrid self-supervised model. Combination of pairwise and contrastive loss.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder()

        # transformations head.
        self.rotation_head = self.get_rotation_head()
        self.jitter_head = self.get_jitter_head()
        self.color_jitter_head = self.get_color_jitter_head()
        self.projection_head = self.get_projection_head()

        # loss weights
        self.loss_weights = self.get_loss_weights()

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

    def get_base_transformation_head(self, output_dim):
        return nn.Sequential(
            nn.Linear(
                self.config.transformation_head_input_dim * 2,
                self.config.transformation_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.transformation_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.transformation_head_hidden_dim, output_dim, bias=False
            ),
        )

    def get_rotation_head(self):
        rotation_head = self.get_base_transformation_head(output_dim=1)
        return rotation_head

    def get_jitter_head(self):
        return self.get_base_transformation_head(output_dim=2)

    def get_color_jitter_head(self):
        return self.get_base_transformation_head(output_dim=4)

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

    def get_loss_weights(self):
        # Dimension should be equal to the loss paramters.
        return nn.Parameter(torch.ones((1, 4)))

    def calculate_loss(self, batch):
        batch_transform1 = batch["transformed_image1"]
        batch_transform2 = batch["transformed_image2"]

        rotation_gt = batch["rotation"]
        jitter_gt = batch["jitter"]
        color_jitter_gt = batch["color_jitter"]

        encoding1 = self.encoder(batch_transform1)
        encoding2 = self.encoder(batch_transform2)

        concat_encoding = torch.cat(
            (self.encoder(batch_transform1), self.encoder(batch_transform2)), 1
        )
        # contrastive projections
        projection1 = nn.functional.normalize(self.projection_head(encoding1))
        projection2 = nn.functional.normalize(self.projection_head(encoding2))

        rotation_pred = self.rotation_head(concat_encoding)
        jitter_pred = self.jitter_head(concat_encoding)
        color_jitter_pred = self.color_jitter_head(concat_encoding)

        # losses

        # contrastive loss
        loss_contrastive = vanila_contrastive_loss(projection1, projection2)

        # regression losses
        loss_rotation = L1Loss()(rotation_gt, rotation_pred)
        loss_jitter = L1Loss()(jitter_gt, jitter_pred)
        loss_color_jitter = L1Loss()(color_jitter_gt, color_jitter_pred)

        loss = torch.sum(
            torch.stack(
                [loss_rotation, loss_jitter, loss_color_jitter, loss_contrastive]
            )
            / torch.abs((self.loss_weights))
            + (torch.log(torch.abs(self.loss_weights)))
        )
        return (
            loss,
            {
                "loss_rotation": loss_rotation.detach(),
                "loss_jitter": loss_jitter.detach(),
                "loss_color_jitter": loss_color_jitter.detach(),
                "loss_contrastive": loss_contrastive.detach(),
                "sigma_rotation": self.loss_weights[0, 0],
                "sigma_jitter": self.loss_weights[0, 1],
                "sigma_color_jitter": self.loss_weights[0, 2],
                "sigma_contrastive": self.loss_weights[0, 3],
            },
            {
                "rotation": [rotation_gt, rotation_pred],
                "jitter": [jitter_gt, jitter_pred],
                "color_jitter": [color_jitter_gt, color_jitter_pred],
            },
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        loss, losses, gt_pred = self.calculate_loss(batch)
        self.train_metrics = {**{"loss": loss.detach()}, **losses}
        self.plot_params = {
            **{
                "image1": batch["transformed_image1"],
                "image2": batch["transformed_image2"],
            },
            **{"gt_pred": gt_pred},
        }
        return {**{"loss": loss}, **losses}

    def training_epoch_end(self, outputs):
        self.train_metrics_epoch = {
            k: torch.stack([x[k] for x in outputs]).mean()
            for k in [
                "loss",
                "loss_rotation",
                "loss_jitter",
                "loss_contrastive",
                "loss_color_jitter",
                "sigma_rotation",
                "sigma_jitter",
                "sigma_color_jitter",
                "sigma_contrastive",
            ]
        }

    def validation_step(self, batch, batch_idx):
        loss, losses, gt_pred = self.calculate_loss(batch)
        self.plot_params = {
            **{
                "image1": batch["transformed_image1"],
                "image2": batch["transformed_image2"],
            },
            **{"gt_pred": gt_pred},
        }
        return {**{"loss": loss}, **losses}

    def validation_epoch_end(self, outputs):

        self.validation_metrics_epoch = {
            k: torch.stack([x[k] for x in outputs]).mean()
            for k in [
                "loss",
                "loss_rotation",
                "loss_jitter",
                "loss_color_jitter",
                "loss_contrastive",
            ]
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
