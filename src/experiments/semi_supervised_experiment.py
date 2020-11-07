import os

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, MASTER_THESIS_DIR, TRAINING_CONFIG_PATH, SSL_CONFIG
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import (
    get_experiement_args,
    prepare_name,
    process_experiment_args,
)
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.utils import get_console_logger, read_json
from torchvision import transforms
from src.models.supervised_head_model import SupervisedHead


def main():
    # get configs
    console_logger = get_console_logger(__name__)
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    args = get_experiement_args()
    train_param, model_param = process_experiment_args(args, console_logger)
    seed_everything(train_param.seed)

    # data preperation
    data = Data_Set(
        config=train_param,
        transform=transforms.Compose([transforms.ToTensor()]),
        train_set=True,
        experiment_type="supervised",
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data,
        batch_size=train_param.batch_size,
        num_workers=train_param.num_workers,
        shuffle=True,
    )
    # Logger

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(DATA_PATH, "models"),
        experiment_name=prepare_name("ssl", train_param),
        disabled=args.testing,
    )

    # model.

    model_param = edict(read_json(SSL_CONFIG))
    model_param.num_samples = len(data)
    model = SupervisedHead(model_param)

    # callbacks
    logging_interval = "step"
    upload_comet_logs = UploadCometLogs(
        logging_interval, get_console_logger("callback"), "supervised"
    )
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    # Trainer setup

    trainer = Trainer(
        accumulate_grad_batches=train_param.accumulate_grad_batches,
        gpus="1" if args.gpu_slow else "0",
        logger=comet_logger,
        max_epochs=train_param.epochs,
        precision=train_param.precision,
        amp_backend="native",
        callbacks=[lr_monitor, upload_comet_logs],
    )
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            MASTER_THESIS_DIR, "src", "models", "supervised_head_model.py"
        ),
    )
    trainer.logger.experiment.log_parameters({"train_param": train_param})
    trainer.logger.experiment.log_parameters({"model_param": model_param})
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
