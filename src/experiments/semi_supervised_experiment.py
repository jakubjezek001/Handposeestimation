import os
from pprint import pformat
from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    SAVED_META_INFO_PATH,
    MASTER_THESIS_DIR,
    TRAINING_CONFIG_PATH,
    SSL_CONFIG,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import (
    get_general_args,
    prepare_name,
    update_train_params,
    downstream_evaluation,
    restore_model,
)
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.utils import get_console_logger, read_json
from torchvision import transforms
from src.models.supervised_head_model import SupervisedHead
from src.models.denoised_supervised_head_model import DenoisedSupervisedHead


def main():
    # get configs
    console_logger = get_console_logger(__name__)
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    args = get_general_args()
    train_param = update_train_params(args, train_param)
    seed_everything(train_param.seed)
    console_logger.info(f"Train parameters {pformat(train_param)}")

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
    experiment_name = prepare_name(f"ssl_{args.experiment_name}", train_param)
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="master-thesis",
        workspace="dahiyaaneesh",
        save_dir=SAVED_META_INFO_PATH,
        experiment_name=experiment_name,
    )

    # model.

    model_param = edict(read_json(SSL_CONFIG))
    model_param.num_samples = len(data)
    model_param.batch_size = train_param.batch_size
    if args.experiment_key is not None:
        model_param.saved_model_name = args.experiment_key
        model_param.checkpoint = args.checkpoint
    model_param.num_of_minibatch = train_param.accumulate_grad_batches
    console_logger.info(f"Model parameters {pformat(model_param)}")
    if args.denoiser:
        model = DenoisedSupervisedHead(model_param)
    else:
        model = SupervisedHead(model_param)

    # callbacks
    logging_interval = args.log_interval
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
            MASTER_THESIS_DIR, "src", "models", "semi_supervised_experiment.py"
        ),
    )

    trainer.logger.experiment.add_tags([args.experiment_type, "SSL", "downstream"])
    trainer.logger.experiment.log_parameters(train_param)
    trainer.logger.experiment.log_parameters(model_param)

    trainer.fit(model, train_data_loader, val_data_loader)

    # restore the best model
    model = restore_model(model, trainer.logger.experiment.get_key())

    # evaluation
    downstream_evaluation(
        model, data, train_param.num_workers, train_param.batch_size, trainer.logger
    )


if __name__ == "__main__":
    main()
