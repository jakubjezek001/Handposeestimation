import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from src.visualization.visualize import plot_truth_vs_prediction


class BaselineModel(LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        if ~self.config["resnet_trainable"]:
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
        if batch_idx == 1:
            if self.config.gpu:
                pred_label = prediction.data[0].numpy()
                try:
                    true_label = y[0].detach().numpy()
                except Exception as e:
                    print(e)
                    true_label = y.data[0].detach().numpy()
            else:
                pred_label = prediction[0].detach().numpy()
                true_label = y[0].detach().numpy()

            plot_truth_vs_prediction(pred_label, true_label, x[0], comet_experiment)
        return {**{"loss": loss}, **train_metrics}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])

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
        distance_joints = (
            torch.sum(((y_pred - y_true) ** 2), 2) ** 0.5
        )  # shape: (batch, 21)
        mean_distance = torch.mean(distance_joints)
        median_distance = torch.median(distance_joints)
        return {
            f"EPE_mean_{step}": mean_distance,
            f"EPE_median_{step}": median_distance,
        }
