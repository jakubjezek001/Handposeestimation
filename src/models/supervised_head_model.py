import math

import torch
import torchvision
from easydict import EasyDict as edict
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from src.models.simclr_model import SimCLR
from src.models.utils import cal_l1_loss, get_latest_checkpoint
from torch import nn


class SupervisedHead(LightningModule):
    """Downstream supervised model to train with encoding from self-supervised models."""

    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder(
            saved_model_path=config.saved_model_name, checkpoint=config.checkpoint
        )
        self.final_layers = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 21 * 3)
        )
        self.train_metrics_epoch = None
        self.train_metrics = None
        self.validation_metrics_epoch = None
        self.plot_params = None

    def get_encoder(self, saved_model_path, checkpoint):
        encoder = torchvision.models.resnet18(pretrained=False)
        encoder.fc = nn.Sequential()
        saved_state_dict = torch.load(
            get_latest_checkpoint(saved_model_path, checkpoint)
        )["state_dict"]
        saved_model_state = {
            key[8:]: value
            for key, value in saved_state_dict.items()
            if "encoder" in key
        }
        encoder.load_state_dict(saved_model_state)
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_layers(x)
        x = x.view(-1, 21, 3)
        return x

    def training_step(self, batch: dict, batch_idx: int):
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

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_z = torch.stack([x["loss_z"] for x in outputs]).mean()
        loss_2d = torch.stack([x["loss_2d"] for x in outputs]).mean()
        loss_z_unscaled = torch.stack([x["loss_z_unscaled"] for x in outputs]).mean()
        self.train_metrics_epoch = {
            "loss": loss.detach(),
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
            "loss_z_unscaled": loss_z_unscaled.detach(),
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

    def validation_epoch_end(self, outputs) -> dict:
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
        self.log("checkpoint_saving_loss", loss)
        self.validation_metrics_epoch = metrics
