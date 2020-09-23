import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from src.visualization.visualize import plot_truth_vs_prediction


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
        train_metrics = self.calculate_metrics(prediction, y, step="train")
        comet_experiment = self.logger.experiment
        comet_experiment.log_metrics({**{"loss": loss}, **train_metrics})
        plot_truth_vs_prediction(
            prediction[0].detach().numpy(), y[0].detach().numpy(), comet_experiment
        )
        return {**{"loss": loss}, **train_metrics}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["joints"]
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        val_metrics = self.calculate_metrics(prediction, y, step="val")
        return {**{"val_loss": loss}, **val_metrics}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_epe_mean = torch.stack([x["EPE_mean_val"] for x in outputs]).mean()
        val_epe_median = torch.stack([x["EPE_median_val"] for x in outputs]).mean()
        self.logger.experiment.log_metrics(
            {
                "val_loss": val_loss,
                "val_epe_mean": val_epe_mean,
                "val_epe_median": val_epe_median,
            }
        )
        return {
            "val_loss": val_loss,
            "val_epe_mean": val_epe_mean,
            "val_epe_median": val_epe_median,
        }

    def calculate_metrics(self, y_pred, y_true, step: str = "train"):
        # Mean distance between predicted and true joint.
        # TODO: Question: Also should these metrics be calculated with prdicted 3D joints or 2.5D predictions?
        # TODO:  Ask Adrian if it is the mean of the median distance between each hand.
        # TODO: Problems in unit of these errors as they are scaled versions.
        # or overall median distance between that batch
        distance_joints = (
            torch.sum(((y_pred - y_true) ** 2), 2) ** 0.5
        )  # shape: (batch, 21)
        mean_distance = torch.mean(distance_joints)
        # mean of the median distances.
        median_distance = torch.mean(torch.median(distance_joints, 1)[0])
        return {
            f"EPE_mean_{step}": mean_distance,
            f"EPE_median_{step}": median_distance,
        }
