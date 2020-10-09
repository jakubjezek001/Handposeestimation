import pytorch_lightning as pl
import torch
import torchvision
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from src.models.utils import log_metrics, vanila_contrastive_loss
from src.utils import get_console_logger
from torch import nn


class SimCLR(LightningModule):
    """
    SimcLR implementation inspired from paper https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    """

    def __init__(self, config):
        super().__init__()
        self.projection_head_input_dim = config.projection_head_input_dim
        self.projection_head_hidden_dim = config.projection_head_hidden_dim
        self.warmup_epochs = config.warmup_epochs
        self.output_dim = config.output_dim
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.opt_weight_decay = config.opt_weight_decay
        # defining model
        self.encoder = self.get_encoder()
        self.projection_head = self.get_projection_head()

    def get_encoder(self):
        encoder = torchvision.models.resnet18(pretrained=True)
        # Removing the last classification layer.
        encoder.fc = nn.Sequential()
        return encoder

    def get_projection_head(self):
        projection_head = nn.Sequential(
            nn.Linear(
                self.projection_head_input_dim,
                self.projection_head_hidden_dim,
                bias=True,
            ),
            nn.ReLU(),
            nn.Linear(self.projection_head_hidden_dim, self.output_dim, bias=False),
        )
        return projection_head

    def contrastive_step(self, batch):
        batch_transform1 = batch["transformed_image1"]
        batch_transform2 = batch["transformed_image2"]
        encoding1 = self.encoder(batch_transform1)
        encoding2 = self.encoder(batch_transform2)
        projection1 = self.projection_head(encoding1)
        projection2 = self.projection_head(encoding2)
        loss = vanila_contrastive_loss(projection1, projection2)
        return loss

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return {"embedding": self.encoder(x), "projection": projection}

    def training_step(self, batch, batch_idx):
        loss = self.contrastive_step(batch)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss)
        # context_val = False
        # comet_logger = self.logger.experiment
        # metrics = {"loss": loss,}
        # log_metrics(metrics, comet_logger, self.current_epoch, context_val)
        return result

    # def validation_step(self, batch, batch_idx):
    #     loss = self.contrastive_step(batch)
    #     return {"loss": loss}

    # def validation_epoch_end(self, outputs):
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     return loss

    def configure_optimizers(self):
        # TODO: understand and add LARS warpper.
        # TODO: Add trick2 for the lr in starting.
        return torch.optim.Adam(self.parameters(), lr=self.lr)
