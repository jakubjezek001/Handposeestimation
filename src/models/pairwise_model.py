import math

import torch
import torchvision
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import MSELoss, L1Loss


class PairwiseModel(LightningModule):
    """
    Pairwise self-supervised model. The transformation parameters are regressed in this
        model.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder()
        self.rotation_head = self.get_rotation_head()
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

    def get_rotation_head(self):
        rotation_head = nn.Sequential(
            nn.Linear(
                self.config.transformation_head_input_dim * 2,
                self.config.transformation_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.transformation_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.transformation_head_hidden_dim,
                self.config.transformation_output_dim.rotation,
                bias=False,
            ),
        )
        return rotation_head

    def transformation_regression_step(self, batch):
        batch_transform1 = batch["transformed_image1"]
        batch_transform2 = batch["transformed_image2"]
        rotation_gt = batch["rotation"]
        encoding = torch.cat(
            (self.encoder(batch_transform1), self.encoder(batch_transform2)), 1
        )
        rotation_pred = self.rotation_head(encoding).view(-1)
        loss = L1Loss()(rotation_gt, rotation_pred)
        return loss, rotation_gt, rotation_pred

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        loss, rotation_gt, rotation_pred = self.transformation_regression_step(batch)
        self.train_metrics = {"loss": loss.detach()}
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "rotation_gt": rotation_gt,
            "rotation_pred": rotation_pred,
        }
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.train_metrics_epoch = {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, rotation_gt, rotation_pred = self.transformation_regression_step(batch)
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "rotation_gt": rotation_gt,
            "rotation_pred": rotation_pred,
        }
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
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
