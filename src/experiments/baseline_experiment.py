import os


import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, MASTER_THESIS_DIR
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import get_experiement_args, process_experiment_args
from src.models.baseline_model import BaselineModel
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.utils import get_console_logger
from torchvision import transforms


def main():
    # get configs

    console_logger = get_console_logger(__name__)
    args = get_experiement_args()
    train_param, model_param = process_experiment_args(args, console_logger)
    seed_everything(train_param.seed)

    # data preperation

    data = Data_Set(
        config=train_param,
        transform=transforms.Compose([transforms.ToTensor()]),
        train_set=True,
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data,
        num_workers=train_param.num_workers,
        batch_size=train_param.batch_size,
        shuffle=True,
    )

    # logger

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(DATA_PATH, "models"),
    )

    # model
    model_param.num_samples = len(data)
    model_param.batch_size = train_param.batch_size
    model = BaselineModel(config=model_param)

    # callbacks

    upload_comet_logs = UploadCometLogs(
        "epoch", get_console_logger("callback"), "supervised"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Training
    if train_param.gpu:
        console_logger.info("GPU Training activated")
        trainer = Trainer(
            max_epochs=train_param.epochs,
            logger=comet_logger,
            precision=train_param.precision,
            amp_backend="native",
            gpus=1,
            callbacks=[lr_monitor, upload_comet_logs],
        )
    else:
        console_logger.info("CPU Training activated")
        trainer = Trainer(
            max_epochs=train_param.epochs,
            logger=comet_logger,
            precision=train_param.precision,
            amp_backend="native",
            callbacks=[lr_monitor, upload_comet_logs],
        )

    trainer.logger.experiment.log_parameters({"train_param": train_param})
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(MASTER_THESIS_DIR, "src", "models", "baseline_model.py"),
    )
    trainer.logger.experiment.log_parameters({"model_param": model_param})
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
