import os

import pytorch_lightning as pl
import torchvision
from easydict import EasyDict as edict
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, FREIHAND_DATA, MASTER_THESIS_DIR
from src.data_loader.freihand_loader2 import F_DB2
from src.data_loader.sample_augmenter import SampleAugmenter
from src.models.simclr_model import SimCLR
from src.utils import read_json
from torch.utils.data import DataLoader


def main():

    augmenter1 = SampleAugmenter(crop=True, resize=True, rotate=True, seed=1)
    augmenter2 = SampleAugmenter(crop=True, resize=True, rotate=True, seed=7)
    data = F_DB2(
        root_dir=os.path.join(FREIHAND_DATA, "training", "rgb"),
        labels_path=os.path.join(FREIHAND_DATA, "training_xyz.json"),
        camera_param_path=os.path.join(FREIHAND_DATA, "training_K.json"),
        transform=torchvision.transforms.ToTensor(),
        augmenter1=augmenter1,
        augmenter2=augmenter2,
    )
    data_loader = DataLoader(data, batch_size=128, num_workers=8)
    # TODO: create suitable validation set.

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
    model_config.num_samples = len(data)
    model = SimCLR(config=model_config)

    # Training
    trainer = pl.Trainer(gpus=[1], logger=comet_logger, max_epochs=100)
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(MASTER_THESIS_DIR, "src", "models", "simclr_model.py"),
    )
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    main()
