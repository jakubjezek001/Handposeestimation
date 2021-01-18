import os
from pprint import pformat

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    COMET_KWARGS,
    HYBRID1_AUGMENTATION_CONFIG,
    HYBRID1_CONFIG,
    HYBRID1_HEATMAP_CONFIG,
    MASTER_THESIS_DIR,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_data, get_train_val_split
from src.experiments.utils import (
    get_callbacks,
    get_hybrid1_args,
    get_model,
    prepare_name,
    save_experiment_key,
    update_hybrid1_train_args,
)
from src.utils import get_console_logger, read_json


def main():
    # get configs
    console_logger = get_console_logger(__name__)
    args = get_hybrid1_args("Hybrid model 1 training script.")

    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param.update(edict(read_json(HYBRID1_AUGMENTATION_CONFIG)))
    train_param = update_hybrid1_train_args(args, train_param)
    model_param_path = HYBRID1_HEATMAP_CONFIG if args.heatmap else HYBRID1_CONFIG
    model_param = edict(read_json(model_param_path))
    console_logger.info(f"Train parameters {pformat(train_param)}")
    seed_everything(train_param.seed)

    data = get_data(
        Data_Set, train_param, sources=args.sources, experiment_type="hybrid1"
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )

    # logger
    experiment_name = prepare_name("hybrid1_", train_param, hybrid_naming=True)
    comet_logger = CometLogger(**COMET_KWARGS, experiment_name=experiment_name)
    # model
    model_param.num_samples = len(data)
    model_param.batch_size = train_param.batch_size
    model_param.num_of_mini_batch = train_param.accumulate_grad_batches
    model_param.contrastive.augmentation = args.contrastive
    model_param.pairwise.augmentation = args.pairwise
    console_logger.info(f"Model parameters {pformat(model_param)}")
    model = get_model(
        experiment_type="hybrid1",
        heatmap_flag=args.heatmap,
        denoiser_flag=args.denoiser,
    )(config=model_param)

    # callbacks
    callbacks = get_callbacks(
        logging_interval=args.log_interval,
        experiment_type="hybrid1",
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
