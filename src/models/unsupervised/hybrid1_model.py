from typing import Dict

import torch
from src.models.base_model import BaseModel
from src.models.unsupervised.pairwise_model import PairwiseModel
from src.models.unsupervised.simclr_model import SimCLR
from torch import nn, Tensor


class Hybrid1Model(SimCLR, PairwiseModel):
    """
    Hybrid self-supervised model. Combination of pairwise and contrastive loss.
    """

    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.config = config
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
        self.plot_params_contrastive = {}
        self.plot_params_pairwise = {}

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Tensor]:
        loss = 0
        losses_all = {}
        self.train_metrics = {}
        if self.make_contrastive_training:
            loss_contrastive = self.contrastive_step(batch["contrastive"])
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
            loss_pairwise, losses, gt_pred = self.transformation_regression_step(
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

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Tensor]:
        loss = 0
        losses_all = {}
        if self.make_contrastive_training:
            loss_contrastive = self.contrastive_step(batch["contrastive"])
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
            loss_pairwise, losses, gt_pred = self.transformation_regression_step(
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
