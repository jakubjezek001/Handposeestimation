import os

from easydict import EasyDict as edict
from numpy.core.numerictypes import _construct_lookups
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, MASTER_THESIS_DIR, NIPS_A1_CONFIG
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import prepare_name, save_experiment_key, get_nips_1a_args
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.models.simclr_model import SimCLR
from src.utils import get_console_logger, read_json
from torchvision import transforms


def main():

    console_logger = get_console_logger(__name__)
    args = get_nips_1a_args()
    console_logger.info(f"Selected_Augmenation: {args.augmentation}")

    # Reading and adjusting configuration parameters.
    params = edict(read_json(NIPS_A1_CONFIG))
    params.data_param.augmentation_flags[args.augmentation] = True
    params.data_param.augmentation_flags.resize = True

    # data preperation
    data = Data_Set(
        config=params.data_param,
        transform=transforms.Compose([transforms.ToTensor()]),
        train_set=True,
        experiment_type="experiment4_pretraining",
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data,
        batch_size=params.data_param.batch_size,
        num_workers=params.data_param.num_workers,
        shuffle=True,
    )

    # Logger
    experiment_name = prepare_name("nips_a1_", params.data_param)
    logging_interval = "epoch"
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=os.path.join(DATA_PATH, "models"),
        experiment_name=experiment_name,
    )

    # model
    params.simclr_param.num_samples = len(data)
    params.simclr_param.batch_size = params.data_param.batch_size
    params.simclr_param.num_of_mini_batch = params.data_param.accumulate_grad_batches
    model = SimCLR(config=params.simclr_param)

    # callbacks

    upload_comet_logs = UploadCometLogs(
        logging_interval, get_console_logger("callback"), experiment_type="simclr"
    )
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    # saves top 3 models in first 100 epochs.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3, period=1, monitor="validate_loss"
    )

    # trainer setup.
    trainer = Trainer(
        accumulate_grad_batches=params.data_param.accumulate_grad_batches,
        gpus="0",
        checkpoint_callback=checkpoint_callback,
        logger=comet_logger,
        max_epochs=params.data_param.epochs,
        precision=params.data_param.precision,
        amp_backend="native",
        callbacks=[lr_monitor, upload_comet_logs],
    )
    # logging information.
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            MASTER_THESIS_DIR, "src", "experiments", "NIPS", "nips_A1_experiment.py"
        ),
    )
    trainer.logger.experiment.add_tags(["NIPS_1A", "SIMCLR", "pretraining"])
    save_experiment_key(
        experiment_name,
        trainer.logger.experiment.get_key(),
        os.path.basename(__file__).replace(".py", ""),
    )
    trainer.logger.experiment.log_parameters(params.data_param)
    trainer.logger.experiment.log_parameters(params.simclr_param)

    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
