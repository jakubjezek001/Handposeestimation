import os
from pprint import pformat

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    SAVED_META_INFO_PATH,
    HYBRID2_CONFIG,
    MASTER_THESIS_DIR,
    TRAINING_CONFIG_PATH,
    COMET_KWARGS,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split, get_data
from src.experiments.utils import (
    get_callbacks,
    get_general_args,
    prepare_name,
    update_train_params,
    save_experiment_key,
)
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.models.hybrid2_model import Hybrid2Model
from src.utils import get_console_logger, read_json
from torchvision import transforms


def main():
    # get configs
    experiment_type = "hybrid2"
    console_logger = get_console_logger(__name__)
    args = get_general_args("Hybrid model 2 training script.")

    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param = update_train_params(args, train_param)
    model_param = edict(read_json(HYBRID2_CONFIG))
    console_logger.info(f"Train parameters {pformat(train_param)}")
    seed_everything(train_param.seed)

    # data preperation
    data = get_data(
        Data_Set, train_param, sources=args.sources, experiment_type=experiment_type
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data,
        batch_size=train_param.batch_size,
        num_workers=train_param.num_workers,
        shuffle=True,
    )
    # Logger
    experiment_name = prepare_name(
        f"{experiment_type}_", train_param, hybrid_naming=False
    )
    comet_logger = CometLogger(**COMET_KWARGS, experiment_name=experiment_name)

    # model
    model_param.num_samples = len(data)
    model_param.batch_size = train_param.batch_size
    model_param.num_of_mini_batch = train_param.accumulate_grad_batches
    model_param.augmentation = [
        key for key, value in train_param.augmentation_flags.items() if value is True
    ]
    console_logger.info(f"Model parameters {pformat(model_param)}")
    model = Hybrid2Model(model_param)

    # callbacks
    callbacks = get_callbacks(
        logging_interval=args.log_interval,
        experiment_type="hybrid2",
        save_top_k=3,
        period=1,
    )
    # trainer
    trainer = Trainer(
        accumulate_grad_batches=train_param.accumulate_grad_batches,
        gpus="0",
        logger=comet_logger,
        max_epochs=train_param.epochs,
        precision=train_param.precision,
        amp_backend="native",
        **callbacks,
    )
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            MASTER_THESIS_DIR, "src", "experiments", "hybrid2_experiment.py"
        ),
    )
    save_experiment_key(
        experiment_name,
        trainer.logger.experiment.get_key(),
        os.path.basename(__file__).replace(".py", ""),
    )
    trainer.logger.experiment.log_parameters(train_param)
    trainer.logger.experiment.log_parameters(model_param)
    trainer.logger.experiment.add_tags(["pretraining", "HYBRID2"] + args.tag)
    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
