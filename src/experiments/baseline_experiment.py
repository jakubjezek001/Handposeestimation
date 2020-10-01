import os
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, MASTER_THESIS_DIR
from src.data_loader.data_set import Data_Set
from src.experiments.utils import get_experiement_args, process_experiment_args
from src.models.baseline_model import BaselineModel
from torch.utils.data import DataLoader
from torchvision import transforms
import copy


def main():
    args = get_experiement_args()
    train_param, model_param = process_experiment_args(args)
    np.random.seed(train_param.seed)
    random.seed(train_param.seed)
    torch.manual_seed(train_param.seed)
    torch.cuda.manual_seed_all(train_param.seed)

    train_data = Data_Set(
        config=train_param,
        transforms=transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        ),
        train_set=True,
    )
    val_data = copy.copy(train_data)
    val_data.is_training(False)

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(DATA_PATH, "models"),
    )
    train_data_loader = DataLoader(
        train_data,
        batch_size=train_param.batch_size,
        num_workers=train_param.num_workers,
    )
    val_data_loader = DataLoader(
        val_data, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )
    model = BaselineModel(config=model_param)
    if train_param.gpu:
        print("GPU Training activated")
        trainer = Trainer(max_epochs=train_param.epochs, logger=comet_logger, gpus=-1)
    else:
        print("CPU Training activated")
        trainer = Trainer(max_epochs=train_param.epochs, logger=comet_logger)
    trainer.logger.experiment.log_parameters({"train_param": train_param})
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(MASTER_THESIS_DIR, "src", "models", "baseline_model.py"),
    )
    trainer.logger.experiment.log_parameters({"model_param": model_param})
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
