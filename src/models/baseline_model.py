import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F


class BaselineModel(LightningModule):
    def __init__(self, freeze_resnet: bool = True):
        super().__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=True)

        if freeze_resnet:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 128)

        self.layer_1 = torch.nn.Linear(128, 128)
        self.output_layer = torch.nn.Linear(128, 21 * 3)

    def forward(self, x):
        batch_size, channel, width, height = x.size()

        x = self.resnet18(x)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = x.view(batch_size, 21, 3)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["joints"]
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
