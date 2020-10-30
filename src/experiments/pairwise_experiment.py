import copy
import os

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, PAIRWISE_CONFIG, TRAINING_CONFIG_PATH
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.models.pairwise_model import PairwiseModel
from src.utils import get_console_logger, read_json
from torchvision import transforms


def main():

    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param.epochs = 100
    train_param.batch_size = 64
    train_param.augmentation_flags = {
        "color_drop": False,
        "color_jitter": True,
        "crop": True,
        "cut_out": False,
        "flip": True,
        "gaussian_blur": True,
        "random_crop": False,
        "resize": True,
        "rotate": True,
    }
    seed_everything(train_param.seed)

    train_data = Data_Set(
        config=train_param,
        transform=transforms.ToTensor(),
        train_set=True,
        experiment_type="pairwise",
    )
    val_data = copy.copy(train_data)
    val_data.is_training(False)

    train_data_loader, val_data_loader = get_train_val_split(
        train_data,
        batch_size=train_param.batch_size,
        num_workers=train_param.num_workers,
        shuffle=False,
    )
    # logger
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(DATA_PATH, "models"),
        experiment_name="pair_rot",
    )
    # model

    model_param = edict(read_json(PAIRWISE_CONFIG))
    model_param.batch_size = train_param.batch_size
    model_param.num_samples = len(train_data)
    model = PairwiseModel(model_param)

    # callbacks
    upload_comet_logs = UploadCometLogs(
        "step", get_console_logger("callback"), "pairwise"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, period=10)
    # trainer
    trainer = Trainer(
        precision=16,
        logger=comet_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[upload_comet_logs, lr_monitor],
        gpus="0",
        max_epochs=train_param.epochs,
    )

    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
