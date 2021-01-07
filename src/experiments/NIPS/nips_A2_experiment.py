import os

from easydict import EasyDict as edict
from numpy.core.numerictypes import _construct_lookups
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import SAVED_META_INFO_PATH, MASTER_THESIS_DIR, NIPS_A2_CONFIG
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import prepare_name, save_experiment_key, get_nips_a2_args
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.models.pairwise_model import PairwiseModel
from src.utils import get_console_logger, read_json
from torchvision import transforms


def main():

    console_logger = get_console_logger(__name__)
    args = get_nips_a2_args()
    console_logger.info(f"Selected_Augmenation: {args.augmentation}")

    # Reading and adjusting configuration parameters.
    params = edict(read_json(NIPS_A2_CONFIG))
    data_param = params.data_param
    pairwise_param = params.pairwise_param
    pairwise_param.augmentation = [args.augmentation]
    data_param.augmentation_flags[args.augmentation] = True
    data_param.augmentation_flags.resize = True

    seed_everything(data_param.seed)
    # data preperation
    data = Data_Set(
        config=data_param,
        transform=transforms.Compose([transforms.ToTensor()]),
        train_set=True,
        experiment_type="pairwise_ablative",
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data,
        batch_size=data_param.batch_size,
        num_workers=data_param.num_workers,
        shuffle=True,
    )

    # Logger
    experiment_name = prepare_name("nips_a2_", data_param)
    logging_interval = "epoch"
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=SAVED_META_INFO_PATH,
        experiment_name=experiment_name,
    )

    # model
    pairwise_param.num_samples = len(data)
    pairwise_param.batch_size = data_param.batch_size
    pairwise_param.num_of_mini_batch = data_param.accumulate_grad_batches
    model = PairwiseModel(config=params.pairwise_param)

    # callbacks

    upload_comet_logs = UploadCometLogs(
        logging_interval, get_console_logger("callback"), experiment_type="pairwise"
    )
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    # saves top 3 models in first 100 epochs.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3, period=1, monitor="checkpoint_saving_loss"
    )

    # trainer setup.
    trainer = Trainer(
        accumulate_grad_batches=data_param.accumulate_grad_batches,
        gpus="0",
        checkpoint_callback=checkpoint_callback,
        logger=comet_logger,
        max_epochs=data_param.epochs,
        precision=data_param.precision,
        amp_backend="native",
        callbacks=[lr_monitor, upload_comet_logs],
    )
    # logging information.
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            MASTER_THESIS_DIR, "src", "experiments", "NIPS", "nips_A2_experiment.py"
        ),
    )
    trainer.logger.experiment.add_tags(["NIPS_A2", "Pairwise", "pretraining"])
    save_experiment_key(
        experiment_name,
        trainer.logger.experiment.get_key(),
        os.path.basename(__file__).replace(".py", ""),
    )
    trainer.logger.experiment.log_parameters(data_param)
    trainer.logger.experiment.log_parameters(params.pairwise_param)

    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
