import logging

from comet_ml import Experiment
from comet_ml import experiment
from pytorch_lightning.callbacks import Callback
from src.models.utils import (
    log_image,
    log_metrics,
    log_simclr_images,
    log_pairwise_images,
)


class UploadCometLogs(Callback):
    """Callback for updating the logs on the comet logger."""

    def __init__(
        self, frequency, console_logger: logging.Logger, experiment_type: str = "simclr"
    ):
        self.frequency = frequency
        self.console_logger = console_logger
        self.valid_logger = False
        self.experiment_type = experiment_type
        if experiment_type == "simclr" or experiment_type == "pairwise":
            self.supervised = False
        else:
            self.supervised = True

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
        if self.valid_logger:
            if self.supervised and batch_idx == 4:
                try:
                    log_image(
                        prediction=pl_module.plot_params["prediction"],
                        y=pl_module.plot_params["ground_truth"],
                        x=pl_module.plot_params["input"],
                        gpu=pl_module.config.gpu,
                        context_val=False,
                        comet_logger=pl_module.logger.experiment,
                    )
                except Exception as e:
                    self.console_logger.error("Unable to upload the images to logger")

                    self.console_logger.info(e)
            elif not self.supervised and batch_idx == 4:
                try:
                    if self.experiment_type == "simclr":
                        log_simclr_images(
                            img1=pl_module.plot_params["image1"],
                            img2=pl_module.plot_params["image2"],
                            context_val=False,
                            comet_logger=pl_module.logger.experiment,
                        )
                    elif self.experiment_type == "pairwise":
                        log_pairwise_images(
                            img1=pl_module.plot_params["image1"],
                            img2=pl_module.plot_params["image2"],
                            rotation_gt=pl_module.plot_params["rotation_gt"],
                            rotation_pred=pl_module.plot_params["rotation_pred"],
                            context_val=False,
                            comet_logger=pl_module.logger.experiment,
                        )
                except Exception as e:
                    self.console_logger.error("Unable to upload the images to logger")
                    self.console_logger.info(e)
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

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.valid_logger:
            if self.supervised and batch_idx == 4:
                try:
                    log_image(
                        prediction=pl_module.plot_params["prediction"],
                        y=pl_module.plot_params["ground_truth"],
                        x=pl_module.plot_params["input"],
                        gpu=pl_module.config.gpu,
                        context_val=True,
                        comet_logger=pl_module.logger.experiment,
                    )
                except Exception as e:
                    self.console_logger.error("Unable to upload the images to logger")
                    self.console_logger.info(e)

            elif not self.supervised and batch_idx == 4:
                try:
                    if self.experiment_type == "simclr":
                        log_simclr_images(
                            img1=pl_module.plot_params["image1"],
                            img2=pl_module.plot_params["image2"],
                            context_val=True,
                            comet_logger=pl_module.logger.experiment,
                        )
                    elif self.experiment_type == "pairwise":
                        log_pairwise_images(
                            img1=pl_module.plot_params["image1"],
                            img2=pl_module.plot_params["image2"],
                            rotation_gt=pl_module.plot_params["rotation_gt"],
                            rotation_pred=pl_module.plot_params["rotation_pred"],
                            context_val=True,
                            comet_logger=pl_module.logger.experiment,
                        )
                except Exception as e:
                    self.console_logger.error("Unable to upload the images to logger")
                    self.console_logger.info(e)
