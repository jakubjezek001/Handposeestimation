import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from src.constants import FREIHAND_DATA
from src.data_loader.freihand_loader import F_DB
from src.experiments.utils import get_experiement_args, process_experiment_args
from src.models.baseline_model import BaselineModel
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    BASE_DIR = os.environ.get("MASTER_THESIS_PATH")
    args = get_experiement_args()
    train_param = process_experiment_args(args)

    f_db = F_DB(
        root_dir=os.path.join(FREIHAND_DATA, "training", "rgb"),
        labels_path=os.path.join(FREIHAND_DATA, "training_xyz.json"),
        camera_param_path=os.path.join(FREIHAND_DATA, "training_K.json"),
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_percentage = int(train_param.train_ratio * 100)
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
    train_data_loader = DataLoader(
        train, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )
    val_data_loader = DataLoader(
        val, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )
    model = BaselineModel(config=train_param)
    if train_param.gpu:
        print("GPU Training ativated")
        trainer = Trainer(max_epochs=train_param.epochs, logger=comet_logger, gpus=-1)
    else:
        print("CPU Training ativated")
        trainer = Trainer(max_epochs=train_param.epochs, logger=comet_logger)
    trainer.logger.experiment.log_parameters(train_param)
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
