from src.models.supervised_head_model import SupervisedHead
from torchvision.models.densenet import DenseNet
from src.models.denoised_baseline import DenoisedBaselineModel
from pytorch_lightning.core.lightning import LightningModule
from easydict import EasyDict as edict
from torch import nn
import torch
from torch.nn import functional as F
import torchvision
from src.models.utils import cal_l1_loss, get_latest_checkpoint


class DenoisedSupervisedHead(DenoisedBaselineModel):
    def __init__(self, config: edict):
        LightningModule.__init__(self)
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

        self.denoiser = nn.Sequential(
            nn.Linear(21 * 3 + 3 * 3 + 1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

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

    # # overloading common methods between both parents
    # def forward(self, x):
    #     return DenoisedBaselineModel.forward(self, x)

    # def training_step(self, batch: dict, batch_idx: int):
    #     return DenoisedBaselineModel.training_step(self, batch, batch_idx)

    # def training_epoch_end(self, outputs):
    #     return DenoisedBaselineModel.training_epoch_end(self, outputs)

    # def setup(self, stage):
    #     return DenoisedBaselineModel.setup(self, stage)

    # def configure_optimizers(self):
    #     return DenoisedBaselineModel.configure_optimizers(self)

    # def validation_step(self, batch, batch_idx):
    #     return DenoisedBaselineModel.validation_step(self, batch, batch_idx)

    # def validation_epoch_end(self, outputs) -> None:
    #     return DenoisedBaselineModel.validation_epoch_end(self, outputs)
