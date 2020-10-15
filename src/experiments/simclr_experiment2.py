import os

import pytorch_lightning as pl
from torchvision import transforms
from easydict import EasyDict as edict
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, MASTER_THESIS_DIR
from src.data_loader.data_set import Data_Set
from src.models.simclr_model import SimCLR
from src.utils import get_console_logger, read_json
from pytorch_lightning.callbacks import LearningRateMonitor
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.constants import TRAINING_CONFIG_PATH
from src.experiments.utils import get_train_val_split


def main():
    train_param = edict(read_json(TRAINING_CONFIG_PATH))

    data = Data_Set(
        config=train_param,
        transform=transforms.Compose([transforms.ToTensor()]),
        train_set=True,
        experiment_type="simclr",
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )
    # Logger

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(DATA_PATH, "models"),
    )

    # model.

    model_config = edict(
        read_json(
            os.path.join(MASTER_THESIS_DIR, "src", "experiments", "simclr_config.json")
        )
    )
    model_config.num_samples = len(train_data_loader)
    model_config.batch_size = 8
    model = SimCLR(config=model_config)

    # callbacks

    upload_comet_logs = UploadCometLogs("epoch", get_console_logger("callback"))
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Training

    trainer = pl.Trainer(
        gpus=0,
        logger=comet_logger,
        max_epochs=100,
        callbacks=[lr_monitor, upload_comet_logs],
    )
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(MASTER_THESIS_DIR, "src", "models", "simclr_model.py"),
    )
    # trainer = pl.Trainer(
    #     gpus=0, max_epochs=100, callbacks=[lr_monitor, upload_comet_logs]
    # )

    trainer.fit(model, train_data_loader)


if __name__ == "__main__":
    main()
