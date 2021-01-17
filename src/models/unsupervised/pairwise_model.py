import math

import torch
import torchvision
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import L1Loss


class PairwiseModel(LightningModule):
    """
    Pairwise self-supervised model. The transformation parameters are regressed in this
        model.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder()
        self.regress_rotate = False
        self.regress_jitter = False
        self.regress_color_jitter = False
        self.log_keys = ["loss"]

        # transformation head.
        if "rotate" in self.config.augmentation:
            self.rotation_head = self.get_rotation_head()
        if "crop" in self.config.augmentation:
            self.jitter_head = self.get_jitter_head()
        if "color_jitter" in self.config.augmentation:
            self.color_jitter_head = self.get_color_jitter_head()

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
        self.regress_rotate = True
        self.log_keys += ["loss_rotation", "sigma_rotation"]
        self.log_sigma_rotate = nn.Parameter(torch.zeros(1, 1))
        rotation_head = self.get_base_transformation_head(output_dim=1)
        return rotation_head

    def get_jitter_head(self):
        self.regress_jitter = True
        self.log_keys += ["loss_jitter", "sigma_jitter"]
        self.log_sigma_jitter = nn.Parameter(torch.zeros(1, 1))
        return self.get_base_transformation_head(output_dim=2)

    def get_color_jitter_head(self):
        self.regress_color_jitter = True
        self.log_keys += ["loss_color_jitter", "sigma_color_jitter"]
        self.log_sigma_color_jitter = nn.Parameter(torch.zeros(1, 1))
        return self.get_base_transformation_head(output_dim=4)

    def regress_rotation(self, rotation_gt, encoding, loss, log: dict, pred_gt):
        rotation_pred = self.rotation_head(encoding)
        loss_rotation = L1Loss()(rotation_gt, rotation_pred)
        loss += loss_rotation / torch.exp(self.log_sigma_rotate) + self.log_sigma_rotate
        log.update(
            {
                "loss_rotation": loss_rotation.detach(),
                "sigma_rotation": torch.exp(self.log_sigma_rotate).detach(),
            }
        )
        pred_gt.update({"rotation": [rotation_gt, rotation_pred]})
        return loss

    def regress_jittering(self, jitter_gt, encoding, loss, log, pred_gt):
        jitter_pred = self.jitter_head(encoding)
        loss_jitter = L1Loss()(jitter_gt, jitter_pred)
        loss += loss_jitter / torch.exp(self.log_sigma_jitter) + self.log_sigma_jitter
        log.update(
            {
                "loss_jitter": loss_jitter.detach(),
                "sigma_jitter": torch.exp(self.log_sigma_jitter).detach(),
            }
        )
        pred_gt.update({"jitter": [jitter_gt, jitter_pred]})
        return loss

    def regress_color_jittering(self, color_jitter_gt, encoding, loss, log, pred_gt):
        color_jitter_pred = self.color_jitter_head(encoding)
        loss_color_jitter = L1Loss()(color_jitter_gt, color_jitter_pred)
        loss += (
            loss_color_jitter / torch.exp(self.log_sigma_color_jitter)
            + self.log_sigma_color_jitter
        )
        log.update(
            {
                "loss_color_jitter": loss_color_jitter.detach(),
                "sigma_color_jitter": torch.exp(self.log_sigma_color_jitter).detach(),
            }
        )
        pred_gt.update({"color_jitter": [color_jitter_gt, color_jitter_pred]})
        return loss

    def transformation_regression_step(self, batch):
        batch_transform1 = batch["transformed_image1"]
        batch_transform2 = batch["transformed_image2"]

        encoding = torch.cat(
            (self.encoder(batch_transform1), self.encoder(batch_transform2)), 1
        )

        loss = 0
        log = {}
        pred_gt = {}
        # Rotation regression
        if self.regress_rotate:
            rotate_gt = batch["rotation"]
            loss = self.regress_rotation(rotate_gt, encoding, loss, log, pred_gt)
        # Translation  jitter regression
        if self.regress_jitter:
            jitter_gt = batch["jitter"]
            loss = self.regress_jittering(jitter_gt, encoding, loss, log, pred_gt)
        # Color jitter regression
        if self.regress_color_jitter:
            color_jitter_gt = batch["color_jitter"]
            loss = self.regress_color_jittering(
                color_jitter_gt, encoding, loss, log, pred_gt
            )
        return (loss, log, pred_gt)

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        loss, losses, gt_pred = self.transformation_regression_step(batch)
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
            k: torch.stack([x[k] for x in outputs]).mean() for k in self.log_keys
        }

    def validation_step(self, batch, batch_idx):
        loss, losses, gt_pred = self.transformation_regression_step(batch)
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
            for k in self.log_keys
            if "loss" in k
        }
        self.log("checkpoint_saving_loss", self.validation_metrics_epoch["loss"])

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
                parameters,
                lr=self.config.lr
                * math.sqrt(self.config.batch_size * self.config.num_of_mini_batch),
            )
        )
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
            "frequency": 1,
        }
        return [optimizer], [scheduler]
