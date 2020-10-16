import argparse
import math
import os

import torch
from easydict import EasyDict as edict
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from src.constants import MASTER_THESIS_DIR, TRAINING_CONFIG_PATH
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.models.simclr_model import SimCLR
from src.models.utils import cal_l1_loss
from src.utils import read_json
from torch import nn
from torchvision import transforms


def main():
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    model_param = edict(
        read_json(
            os.path.join(MASTER_THESIS_DIR, "src", "experiments", "simclr_config.json")
        )
    )
    train_param.augmentation_flags.resize = True
    train_param.augmentation_flags.rotate = True
    train_param.augmentation_flags.crop = True

    data = Data_Set(
        config=train_param,
        transform=transforms.Compose([transforms.ToTensor()]),
        train_set=True,
        experiment_type="supervised",
    )
    model_param.num_samples = len(data)
    model_param.alpha = 5
    model_param.gpu = True

    train_data_loader, val_data_loader = get_train_val_split(
        data, batch_size=2048, num_workers=train_param.num_workers
    )

    with torch.cuda.amp.autocast():
        model_ssl = SupervisedHead(
            model_param,
            "/local/home/adahiya/Documents/master_thesis/data/models/master-thesis/a7c43b88d9c34332bf3c86eb81f5db7d/checkpoints/epoch=99.ckpt",
            model_param,
        )

    trainer = Trainer(precision=16, gpus=1, amp_backend="native", max_epochs=2)
    trainer.fit(model_ssl, train_data_loader, val_data_loader)


class SupervisedHead(LightningModule):
    def __init__(
        self, simclr_config: edict, saved_simclr_model_path: str, config: edict
    ):
        super().__init__()
        self.config = config
        self.encoder = self.get_simclr_model(simclr_config, saved_simclr_model_path)
        self.temp_layer = nn.Sigmoid()
        self.final_layers = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 21 * 3)
        )
        self.train_metrics_epoch = None
        self.train_metrics = None
        self.validation_metrics_epoch = None
        self.plot_params = None

    def get_simclr_model(self, simclr_config, saved_simclr_model_path):
        simclr_model = SimCLR(simclr_config)
        saved_model_state = torch.load(saved_simclr_model_path)["state_dict"]
        simclr_model.load_state_dict(saved_model_state)
        for param in simclr_model.parameters():
            param.requires_grad = False
        return simclr_model.encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.temp_layer(x)
        x = self.final_layers(x)
        x = x.view(-1, 21, 3)
        return x

    def training_step(self, batch: dict, batch_idx: int):
        # with autocast(True):
        x, y = batch["image"], batch["joints"]
        prediction = self(x)
        loss_2d, loss_z = cal_l1_loss(prediction, y)
        loss = loss_2d + self.config.alpha * loss_z
        metrics = {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}
        self.train_metrics = metrics
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}
        return metrics

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_z = torch.stack([x["loss_z"] for x in outputs]).mean()
        loss_2d = torch.stack([x["loss_2d"] for x in outputs]).mean()
        self.train_metrics_epoch = {"loss": loss, "loss_z": loss_z, "loss_2d": loss_2d}

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
        self.config.warmup_epochs = (
            self.config.warmup_epochs * self.train_iters_per_epoch
        )
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config.warmup_epochs,
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
        # with autocast(True):
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


if __name__ == "__main__":
    main()
