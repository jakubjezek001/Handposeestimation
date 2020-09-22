import os
import sys

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

    BASE_DIR = "/home/aneesh/Documents/master_thesis/"
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(BASE_DIR, "models"),
    )
    f_db = F_DB(
        root_dir=os.path.join(BASE_DIR, "data/raw/FreiHAND_pub_v2/training/rgb"),
        labels_path=os.path.join(
            BASE_DIR, "data/raw/FreiHAND_pub_v2/training_xyz.json"
        ),
        camera_param_path=os.path.join(
            BASE_DIR, "data/raw/FreiHAND_pub_v2/training_K.json"
        ),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        gray=False,
    )
    data_loader = DataLoader(f_db, batch_size=16)
    model = BaselineModel()
    trainer = Trainer(max_epochs=3, logger=comet_logger)
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    main()
