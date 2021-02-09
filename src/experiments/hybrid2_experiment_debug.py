import os
from pprint import pformat


from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    COMET_KWARGS,
    HYBRID2_CONFIG,
    HYBRID2_HEATMAP_CONFIG,
    MASTER_THESIS_DIR,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split, get_data
from src.experiments.utils import (
    get_callbacks,
    get_general_args,
    get_model,
    prepare_name,
    save_experiment_key,
    update_model_params,
    update_train_params,
)
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.models.unsupervised.hybrid2_model import Hybrid2Model
from src.utils import get_console_logger, read_json
from torchvision import transforms


def main():
    # get configs
    console_logger = get_console_logger(__name__)
    args = get_general_args("Hybrid model 2 training script.")

    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param = update_train_params(args, train_param)
    model_param = edict(read_json(HYBRID2_CONFIG))
    console_logger.info(f"Train parameters {pformat(train_param)}")
    seed_everything(train_param.seed)
    experiment_type = "hybrid2"
    data = get_data(
        Data_Set, train_param, sources=args.sources, experiment_type=experiment_type
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )

    # Logger
    # experiment_name = prepare_name("hybrid2_", train_param, hybrid_naming=False)
    # comet_logger = CometLogger(
    #     api_key=os.environ.get("COMET_API_KEY"),
    #     project_name="master-thesis",
    #     workspace="dahiyaaneesh",
    #     save_dir=SAVED_META_INFO_PATH,
    #     experiment_name=experiment_name,
    # )

    # model
    model_param = update_model_params(model_param, args, len(data), train_param)
    model_param.augmentation = [
        key for key, value in train_param.augmentation_flags.items() if value
    ]
    console_logger.info(f"Model parameters {pformat(model_param)}")
    model = get_model(
        experiment_type="hybrid2",
        heatmap_flag=args.heatmap,
        denoiser_flag=args.denoiser,
    )(config=model_param)

    # callbacks
    # logging_interval = "epoch"
    # upload_comet_logs = UploadCometLogs(
    #     logging_interval, get_console_logger("callback"), "hybrid2"
    # )
    # lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    # saving the best model as per the validation loss.
    # checkpoint_callback = ModelCheckpoint(
    #     save_top_k=3, period=1, monitor="checkpoint_saving_loss"
    # )

    # trainer
    trainer = Trainer(
        accumulate_grad_batches=train_param.accumulate_grad_batches,
        gpus="0",
        # checkpoint_callback=checkpoint_callback,
        # logger=comet_logger,
        max_epochs=train_param.epochs,
        precision=train_param.precision,
        amp_backend="native",
        # callbacks=[lr_monitor, upload_comet_logs],
    )
    # trainer.logger.experiment.set_code(
    #     overwrite=True,
    #     filename=os.path.join(
    #         MASTER_THESIS_DIR, "src", "experiments", "hybrid2_experiment.py"
    #     ),
    # )
    # save_experiment_key(
    #     experiment_name,
    #     trainer.logger.experiment.get_key(),
    #     os.path.basename(__file__).replace(".py", ""),
    # )
    # trainer.logger.experiment.log_parameters(train_param)
    # trainer.logger.experiment.log_parameters(model_param)
    # trainer.logger.experiment.add_tags(["pretraining", "HYBRID2"] + args.tag)
    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
