import math
from typing import List

import torch
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule


class BaseModel(LightningModule):
    """This is the base class inherited by all the models used in the thesis.
    It on the other hand inherits from Lightening module. It defines functions for
    setting up optimizer, schedulers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_metrics_epoch = None
        self.train_metrics = None
        self.validation_metrics_epoch = None
        self.plot_params = None

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list: List[str] = ["bias", "bn"]
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

    def setup(self, stage: str):
        global_batch_size = self.trainer.world_size * self.config.batch_size
        self.train_iters_per_epoch = self.config.num_samples // global_batch_size

    def configure_optimizers(self):
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.config.opt_weight_decay
        )
        if self.config.optimizer == "LARS":
            optimizer = LARSWrapper(
                torch.optim.Adam(
                    parameters, lr=self.config.lr * math.sqrt(self.config.batch_size)
                )
            )
        else:
            optimizer = torch.optim.Adam(
                parameters, lr=self.config.lr * math.sqrt(self.config.batch_size)
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

    def training_epoch_end(self, outputs: List[dict]):
        metric_keys = outputs[0].keys()
        self.train_metrics_epoch = {
            key: torch.stack([x[key] for x in outputs]).mean() for key in metric_keys
        }

    def validation_epoch_end(self, outputs: List[dict]):
        metric_keys = outputs[0].keys()
        metrics = {
            key: torch.stack([x[key] for x in outputs]).mean() for key in metric_keys
        }
        self.log("checkpoint_saving_loss", metrics["loss"])
        self.validation_metrics_epoch = metrics
