import math

import torch
import torchvision
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import L1Loss, CrossEntropyLoss
from src.models.utils import vanila_contrastive_loss


class Hybrid1Model(LightningModule):
    """
    Hybrid self-supervised model. Combination of pairwise and contrastive loss.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder()
        self.regress_rotate = False
        self.regress_jitter = False
        self.regress_color_jitter = False
        self.log_keys = ["loss"]
        try:
            self.config.pairwise.augmentation.remove("resize")
        except Exception as e:
            print(e)
        try:
            self.config.contrastive.augmentation.remove("resize")
        except Exception as e:
            print(e)
        # transformations head.
        if len(self.config.pairwise.augmentation) != 0:
            self.make_pairwise_training = True
            self.log_keys += ["loss_pairwise", "sigma_pairwise"]
            self.log_sigma_pairwise = nn.Parameter(torch.zeros(1, 1))
            if "rotate" in self.config.pairwise.augmentation:
                self.rotation_head = self.get_rotation_head()
            if "crop" in self.config.pairwise.augmentation:
                self.jitter_head = self.get_jitter_head()
            if "color_jitter" in self.config.pairwise.augmentation:
                self.color_jitter_head = self.get_color_jitter_head()
        else:
            self.make_pairwise_training = False

        if len(self.config.contrastive.augmentation) != 0:
            self.log_sigma_contrastive = nn.Parameter(torch.zeros(1, 1))
            self.make_contrastive_training = True
            self.log_keys += ["loss_contrastive", "sigma_contrastive"]
            self.projection_head = self.get_projection_head()
        else:
            self.make_contrastive_training = False

        # Variables used by callbacks
        self.train_metrics_epoch = None
        self.train_metrics = None
        self.validation_metrics_epoch = None
        self.plot_params_contrastive = None
        self.plot_params_pairwise = None

    def get_encoder(self):
        encoder = torchvision.models.resnet18(pretrained=True)
        # Removing the last classification layer.
        encoder.fc = nn.Sequential()
        return encoder

    def get_base_transformation_head(self, output_dim):
        return nn.Sequential(
            nn.Linear(
                self.config.pairwise.transformation_head_input_dim * 2,
                self.config.pairwise.transformation_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.pairwise.transformation_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.pairwise.transformation_head_hidden_dim,
                output_dim,
                bias=False,
            ),
        )

    # Regression heads
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

    # Contrastive projections head.
    def get_projection_head(self):
        projection_head = nn.Sequential(
            nn.Linear(
                self.config.contrastive.projection_head_input_dim,
                self.config.contrastive.projection_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.contrastive.projection_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.contrastive.projection_head_hidden_dim,
                self.config.contrastive.output_dim,
                bias=False,
            ),
        )
        return projection_head

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

    def calculate_contrastive_loss(self, batch):
        batch_transform1 = batch["transformed_image1"]
        batch_transform2 = batch["transformed_image2"]
        encoding1 = self.encoder(batch_transform1)
        encoding2 = self.encoder(batch_transform2)
        # contrastive projections
        projection1 = nn.functional.normalize(self.projection_head(encoding1))
        projection2 = nn.functional.normalize(self.projection_head(encoding2))
        loss = vanila_contrastive_loss(projection1, projection2)
        return loss

    def calculate_pairwise_loss(self, batch):
        loss = 0
        log = {}
        pred_gt = {}
        concat_encoding = torch.cat(
            (
                self.encoder(batch["transformed_image1"]),
                self.encoder(batch["transformed_image2"]),
            ),
            1,
        )
        # Rotation regression
        if self.regress_rotate:
            rotate_gt = batch["rotation"]
            loss = self.regress_rotation(rotate_gt, concat_encoding, loss, log, pred_gt)

        # Translation  jitter regression
        if self.regress_jitter:
            jitter_gt = batch["jitter"]
            loss = self.regress_jittering(
                jitter_gt, concat_encoding, loss, log, pred_gt
            )
        # Color jitter regression
        if self.regress_color_jitter:
            color_jitter_gt = batch["color_jitter"]
            loss = self.regress_color_jittering(
                color_jitter_gt, concat_encoding, loss, log, pred_gt
            )

        log.update({"loss_pairwise": loss.detach()})
        return (loss, log, pred_gt)

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        loss = 0
        losses_all = {}
        self.train_metrics = {}
        if self.make_contrastive_training:
            loss_contrastive = self.calculate_contrastive_loss(batch["contrastive"])
            loss += (
                loss_contrastive / torch.exp(self.log_sigma_contrastive)
                + self.log_sigma_contrastive
            )
            self.train_metrics = {
                "loss_contrastive": loss_contrastive.detach(),
                "sigma_contrastive": torch.exp(self.log_sigma_contrastive).detach(),
            }
            losses_all = self.train_metrics
            self.plot_params_contrastive = {
                **{
                    "image1": batch["contrastive"]["transformed_image1"],
                    "image2": batch["contrastive"]["transformed_image2"],
                }
            }

        if self.make_pairwise_training:
            loss_pairwise, losses, gt_pred = self.calculate_pairwise_loss(
                batch["pairwise"]
            )
            self.train_metrics.update(losses)
            self.train_metrics.update(
                {
                    "loss_pairwise": loss_pairwise.detach(),
                    "sigma_pairwise": torch.exp(self.log_sigma_pairwise).detach(),
                }
            )
            losses_all.update(losses)
            losses_all.update(
                {"sigma_pairwise": torch.exp(self.log_sigma_pairwise).detach()}
            )
            self.plot_params_pairwise = {
                **{
                    "image1": batch["pairwise"]["transformed_image1"],
                    "image2": batch["pairwise"]["transformed_image2"],
                },
                **{"gt_pred": gt_pred},
            }
            loss += (
                torch.squeeze(loss_pairwise) / torch.exp(self.log_sigma_pairwise)
                + self.log_sigma_pairwise
            )

        self.train_metrics = {**{"loss": loss.detach()}, **self.train_metrics}

        return {**{"loss": loss}, **losses_all}

    def training_epoch_end(self, outputs):
        self.train_metrics_epoch = {
            k: torch.stack([x[k] for x in outputs]).mean() for k in self.log_keys
        }

    def validation_step(self, batch, batch_idx):
        loss = 0
        losses_all = {}
        if self.make_contrastive_training:
            loss_contrastive = self.calculate_contrastive_loss(batch["contrastive"])
            loss += (
                loss_contrastive / torch.exp(self.log_sigma_contrastive)
                + self.log_sigma_contrastive
            )
            losses_all = {
                "loss_contrastive": loss_contrastive.detach(),
                "sigma_contrastive": torch.exp(self.log_sigma_contrastive).detach(),
            }
            self.plot_params_contrastive = {
                **{
                    "image1": batch["contrastive"]["transformed_image1"],
                    "image2": batch["contrastive"]["transformed_image2"],
                }
            }

        if self.make_pairwise_training:
            loss_pairwise, losses, gt_pred = self.calculate_pairwise_loss(
                batch["pairwise"]
            )
            losses_all.update(losses)
            losses_all.update(
                {"sigma_pairwise": torch.exp(self.log_sigma_pairwise).detach()}
            )
            self.plot_params_pairwise = {
                **{
                    "image1": batch["pairwise"]["transformed_image1"],
                    "image2": batch["pairwise"]["transformed_image2"],
                },
                **{"gt_pred": gt_pred},
            }
            loss += (
                torch.squeeze(loss_pairwise) / torch.exp(self.log_sigma_pairwise)
                + self.log_sigma_pairwise
            )

        return {**{"loss": loss}, **losses_all}

    def validation_epoch_end(self, outputs):
        self.validation_metrics_epoch = {
            k: torch.stack([x[k] for x in outputs]).mean() for k in self.log_keys
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
