import os

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import DATA_PATH, MASTER_THESIS_DIR, DOWNSTREAM_CONFIG
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import (
    get_downstream_args,
    prepare_name,
    process_experiment_args,
    downstream_evaluation,
    restore_model,
    save_experiment_key,
    get_checkpoints,
)
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.utils import get_console_logger, read_json
from torchvision import transforms
from src.models.supervised_head_model import SupervisedHead


def main():
    # get configs
    console_logger = get_console_logger(__name__)
    params = edict(read_json(DOWNSTREAM_CONFIG))
    data_param = params.data_param
    args = get_downstream_args()
    console_logger.info(
        "Downstream experiment will be run for"
        f" {len(get_checkpoints(args.experiment_key))} checkpoints"
    )
    for checkpoint in get_checkpoints(args.experiment_key):
        seed_everything(data_param.seed)
        console_logger.info(f"Checkpoint {args.experiment_name} {checkpoint}")
        # data preperation

        data = Data_Set(
            config=data_param,
            transform=transforms.Compose([transforms.ToTensor()]),
            train_set=True,
            experiment_type="supervised",
        )
        train_data_loader, val_data_loader = get_train_val_split(
            data,
            batch_size=data_param.batch_size,
            num_workers=data_param.num_workers,
            shuffle=True,
        )
        # Logger
        experiment_name = prepare_name(f"ssl_{args.experiment_name}", data_param)
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="master-thesis",
            workspace="dahiyaaneesh",
            save_dir=os.path.join(DATA_PATH, "models"),
            experiment_name=experiment_name,
        )

        # Model.
        model_param = params.model_param
        model_param.num_samples = len(data)
        model_param.saved_model_name = args.experiment_key
        model_param.checkpoint = checkpoint
        model_param.batch_size = data_param.batch_size
        model = SupervisedHead(model_param)

        # callbacks
        logging_interval = "epoch"
        upload_comet_logs = UploadCometLogs(
            logging_interval, get_console_logger("callback"), "supervised"
        )
        lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
        # saving the best model as per the validation loss.

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, period=1, monitor="checkpoint_saving_loss"
        )
        # Trainer setup

        trainer = Trainer(
            accumulate_grad_batches=data_param.accumulate_grad_batches,
            gpus="0",
            logger=comet_logger,
            max_epochs=data_param.epochs,
            precision=data_param.precision,
            amp_backend="native",
            callbacks=[lr_monitor, upload_comet_logs],
            checkpoint_callback=checkpoint_callback,
        )
        trainer.logger.experiment.set_code(
            overwrite=True,
            filename=os.path.join(
                MASTER_THESIS_DIR,
                "src",
                "experiments",
                "NIPS",
                "downstream_experiment.py",
            ),
        )
        trainer.logger.experiment.add_tags([args.experiment_type, "SSL", "downstream"])
        save_experiment_key(
            experiment_name,
            trainer.logger.experiment.get_key(),
            f"{args.experiment_type.lower()}_downstream",
        )
        trainer.logger.experiment.log_parameters(data_param)
        trainer.logger.experiment.log_parameters(model_param)

        trainer.fit(model, train_data_loader, val_data_loader)

        # restore the best model
        model = restore_model(model, trainer.logger.experiment.get_key())

        # evaluation
        downstream_evaluation(
            model, data, data_param.num_workers, data_param.batch_size, trainer.logger
        )


if __name__ == "__main__":
    main()
