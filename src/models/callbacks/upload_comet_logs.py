import logging

from comet_ml import Experiment
from pytorch_lightning.callbacks import Callback
from src.models.utils import log_metrics


class UploadCometLogs(Callback):
    """Callback for updating the logs on the comet logger."""

    def __init__(self, frequency, console_logger: logging.Logger):
        self.frequency = frequency
        self.console_logger = console_logger
        self.valid_logger = False

    def on_fit_start(self, trainer, pl_module):
        if isinstance(pl_module.logger.experiment, Experiment):
            self.valid_logger = True
        else:
            self.console_logger.error(
                "Comet logger object missing! Logs won't be updated"
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.valid_logger and self.frequency == "step":
            try:
                log_metrics(
                    metrics=pl_module.train_metrics,
                    comet_logger=pl_module.logger.experiment,
                    epoch=pl_module.current_epoch,
                    context_val=False,
                )
            except Exception as e:
                self.console_logger.error("Unable to upload the metrics to logger")
                self.console_logger.info(e)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if self.valid_logger and self.frequency == "epoch":
            try:
                log_metrics(
                    metrics=pl_module.train_metrics_epoch,
                    comet_logger=pl_module.logger.experiment,
                    epoch=pl_module.current_epoch,
                    context_val=False,
                )
            except Exception as e:
                self.console_logger.error("Unable to upload the metrics to logger")
                self.console_logger.info(e)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.valid_logger:
            try:
                log_metrics(
                    metrics=pl_module.validation_metrics_epoch,
                    comet_logger=pl_module.logger.experiment,
                    epoch=pl_module.current_epoch,
                    context_val=True,
                )

            except Exception as e:
                self.console_logger.error(
                    "Unable to upload the validation metrics to logger"
                )
                self.console_logger.info(e)
