import os
import sys

import torch
import pytorch_lightning as pl
from comet_ml import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    sys.path.append("/home/aneesh/Documents/master_thesis")
    from src.data_loader.freihand_loader import F_DB
    from src.data_loader.utils import convert_2_5D_to_3D
    from src.models.baseline_model import BaselineModel
    from src.constants import FREIHAND_DATA

    BASE_DIR = "/home/aneesh/Documents/master_thesis/"
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="baseline",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(BASE_DIR, "models"),
    )
    f_db = F_DB(
        root_dir=os.path.join(FREIHAND_DATA, "training", "rgb"),
        labels_path=os.path.join(FREIHAND_DATA, "training_xyz.json"),
        camera_param_path=os.path.join(FREIHAND_DATA, "training_K.json"),
        transform=transforms.Compose([transforms.ToTensor()]),
        gray=False,
    )
    train_percentage = 90
    train, val = torch.utils.data.random_split(
        f_db,
        [
            len(f_db) * train_percentage // 100,
            len(f_db) - len(f_db) * train_percentage // 100,
        ],
    )
    train_data_loader = DataLoader(train, batch_size=16)
    val_data_loader = DataLoader(val, batch_size=4)
    model = BaselineModel()
    trainer = Trainer(max_epochs=10, logger=comet_logger)
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
