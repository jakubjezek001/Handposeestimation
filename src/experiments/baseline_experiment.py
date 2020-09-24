import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from src.constants import FREIHAND_DATA
from src.data_loader.freihand_loader import F_DB
from src.experiments.utils import get_experiement_args
from src.models.baseline_model import BaselineModel
from src.utils import read_json
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    BASE_DIR = os.environ.get("MASTER_THESIS_PATH")
    gpu_use = get_experiement_args()
    training_hyper_param = read_json(
        os.path.join(BASE_DIR, "src", "experiments", "training_config.json")
    )

    f_db = F_DB(
        root_dir=os.path.join(FREIHAND_DATA, "training", "rgb"),
        labels_path=os.path.join(FREIHAND_DATA, "training_xyz.json"),
        camera_param_path=os.path.join(FREIHAND_DATA, "training_K.json"),
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_percentage = int(training_hyper_param["train_ratio"] * 100)
    train, val = torch.utils.data.random_split(
        f_db,
        [
            len(f_db) * train_percentage // 100,
            len(f_db) - len(f_db) * train_percentage // 100,
        ],
    )
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(BASE_DIR, "models"),
    )
    train_data_loader = DataLoader(train, batch_size=training_hyper_param["batch_size"])
    val_data_loader = DataLoader(val, batch_size=training_hyper_param["batch_size"])
    model = BaselineModel(freeze_resnet=training_hyper_param["resnet_trainable"])
    if gpu_use:
        print("GPU Training ativated")
        trainer = Trainer(
            max_epochs=training_hyper_param["epochs"], logger=comet_logger, gpus=-1
        )
    else:
        print("CPU Training ativated")
        trainer = Trainer(
            max_epochs=training_hyper_param["epochs"], logger=comet_logger
        )
    trainer.logger.experiment.log_parameters(training_hyper_param)
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
