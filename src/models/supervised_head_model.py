import math

import torch
from easydict import EasyDict as edict
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from src.models.simclr_model import SimCLR
from src.models.utils import cal_l1_loss, get_latest_checkpoint
from torch import nn


class SupervisedHead(LightningModule):
    """Downstream supervised model to train with encoding from self-supervised models."""

    def __init__(self, simclr_config: edict, config: edict):
        super().__init__()
        self.config = config
        self.encoder = self.get_simclr_model(
            simclr_config, config.simclr_experiment_name, config.checkpoint
        )
        self.final_layers = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 21 * 3)
        )
        self.train_metrics_epoch = None
        self.train_metrics = None
        self.validation_metrics_epoch = None
        self.plot_params = None

    def get_simclr_model(self, simclr_config, saved_simclr_model_path, checkpoint):
        simclr_model = SimCLR(simclr_config)
        saved_model_state = torch.load(
            get_latest_checkpoint(saved_simclr_model_path, checkpoint)
        )["state_dict"]
        simclr_model.load_state_dict(saved_model_state)
        # simclr_model.eval()
        for param in simclr_model.parameters():
            param.requires_grad = False
        return simclr_model.encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_layers(x)
        x = x.view(-1, 21, 3)
        return x

    def training_step(self, batch: dict, batch_idx: int):
        x, y = batch["image"], batch["joints"]
        prediction = self(x)
        loss_2d, loss_z = cal_l1_loss(prediction, y)
        loss = loss_2d + self.config.alpha * loss_z
        self.train_metrics = {
            "loss": loss.detach(),
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
        }
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}
        return {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_z = torch.stack([x["loss_z"] for x in outputs]).mean()
        loss_2d = torch.stack([x["loss_2d"] for x in outputs]).mean()
        self.train_metrics_epoch = {
            "loss": loss.detach(),
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
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
        x, y = batch["image"], batch["joints"]
        prediction = self(x)
        loss_2d, loss_z = cal_l1_loss(prediction, y)
        loss = loss_2d + self.config.alpha * loss_z
        metrics = {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}
        return metrics

    def validation_epoch_end(self, outputs) -> dict:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_z = torch.stack([x["loss_z"] for x in outputs]).mean()
        loss_2d = torch.stack([x["loss_2d"] for x in outputs]).mean()
        metrics = {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}
        self.validation_metrics_epoch = metrics
