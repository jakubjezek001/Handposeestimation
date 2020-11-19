import math

import torch
import torchvision
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from src.models.utils import vanila_contrastive_loss
from torch import nn
from torch.nn import functional as F


class SimCLR(LightningModule):
    """
    SimcLR implementation inspired from paper https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
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

    def contrastive_step(self, batch):
        batch_transform1 = batch["transformed_image1"]
        batch_transform2 = batch["transformed_image2"]
        encoding1 = self.encoder(batch_transform1)
        encoding2 = self.encoder(batch_transform2)
        projection1 = F.normalize(self.projection_head(encoding1))
        projection2 = F.normalize(self.projection_head(encoding2))
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
        }
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, log=True)
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
                parameters, lr=self.config.lr * math.sqrt(self.config.batch_size)
            )
        )

        # The schdeuler is called after every step in an epoch hence adjusting the
        # warmup epochs param.
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
