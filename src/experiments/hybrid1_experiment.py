import os
from pprint import pformat

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    DATA_PATH,
    HYBRID1_CONFIG,
    HYBRID1_AUGMENTATION_CONFIG,
    MASTER_THESIS_DIR,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import (
    get_hybrid1_args,
    prepare_name,
    update_hybrid1_train_args,
    save_experiment_key,
)
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.models.hybrid1_model import Hybrid1Model
from src.utils import get_console_logger, read_json
from torchvision import transforms


def main():
    # get configs
    console_logger = get_console_logger(__name__)
    args = get_hybrid1_args("Hybrid model 1 training script.")

    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param.update(edict(read_json(HYBRID1_AUGMENTATION_CONFIG)))
    train_param.pairwise.augmentation_flags.resize = True
    train_param.contrastive.augmentation_flags.resize = True
    train_param = update_hybrid1_train_args(args, train_param)
    model_param = edict(read_json(HYBRID1_CONFIG))
    console_logger.info(f"Train parameters {pformat(train_param)}")
    seed_everything(train_param.seed)

    data = Data_Set(
        config=train_param,
        transform=transforms.ToTensor(),
        train_set=True,
        experiment_type="hybrid1",
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data,
        batch_size=train_param.batch_size,
        num_workers=train_param.num_workers,
        shuffle=True,
    )

    # logger
    experiment_name = prepare_name("hybrid1_", train_param, hybrid_naming=True)
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(DATA_PATH, "models"),
        experiment_name=experiment_name,
    )
    # model
    model_param.num_samples = len(data)
    model_param.batch_size = train_param.batch_size
    model_param.num_of_mini_batch = train_param.accumulate_grad_batches
    model_param.contrastive.augmentation = args.contrastive
    model_param.pairwise.augmentation = args.pairwise
    console_logger.info(f"Model parameters {pformat(model_param)}")
    model = Hybrid1Model(model_param)

    # callbacks
    logging_interval = "epoch"
    upload_comet_logs = UploadCometLogs(
        logging_interval, get_console_logger("callback"), "hybrid1"
    )
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    # saving the best model as per the validation loss.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, period=1, monitor="checkpoint_saving_loss"
    )

    # trainer
    trainer = Trainer(
        accumulate_grad_batches=train_param.accumulate_grad_batches,
        gpus="0",
        logger=comet_logger,
        max_epochs=train_param.epochs,
        precision=train_param.precision,
        amp_backend="native",
        callbacks=[lr_monitor, upload_comet_logs],
        checkpoint_callback=checkpoint_callback,
    )
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            MASTER_THESIS_DIR, "src", "experiments", "hybrid1_experiment.py"
        ),
    )
    save_experiment_key(
        experiment_name,
        trainer.logger.experiment.get_key(),
        os.path.basename(__file__).replace(".py", ""),
    )
    trainer.logger.experiment.log_parameters(train_param)
    trainer.logger.experiment.log_parameters(model_param)
    trainer.logger.experiment.add_tags(["pretraining", "HYBRID1"] + args.tag)
    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
