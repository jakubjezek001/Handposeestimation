import logging

from comet_ml import Experiment
from pytorch_lightning.callbacks import Callback
from src.models.utils import log_metrics


class UploadCometLogs(Callback):
    def __init__(self, frequency, console_logger: logging.Logger):
        self.frequency = frequency
        self.console_logger = console_logger

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not isinstance(pl_module.logger.experiment, Experiment):
            return

        if self.frequency == "step":
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
        if not isinstance(pl_module.logger.experiment, Experiment):
            return

        if self.frequency == "epoch":
            try:
                log_metrics(
                    metrics=pl_module.train_metric_epoch,
                    comet_logger=pl_module.logger.experiment,
                    epoch=pl_module.current_epoch,
                    context_val=False,
                )
            except Exception as e:
                self.console_logger.error("Unable to upload the metrics to logger")
                self.console_logger.info(e)
