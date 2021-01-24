import os

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    SAVED_META_INFO_PATH,
    MASTER_THESIS_DIR,
    TRAINING_CONFIG_PATH,
    SSL_CONFIG,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.experiments.utils import prepare_name
from src.models.callbacks.upload_comet_logs import UploadCometLogs
from src.utils import get_console_logger, read_json
from torchvision import transforms
from src.models.semisupervised.supervised_head_model import SupervisedHead
import argparse
from src.experiments.evaluation_utils import calculate_epe_statistics, evaluate


def main():
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    parser = argparse.ArgumentParser(description="Script for Experiement 2bc")
    parser.add_argument("-checkpoint", type=int, help="Epoch number")
    args = parser.parse_args()
    train_param.augmentation_flags = {
        "color_drop": False,
        "color_jitter": False,
        "crop": True,
        "cut_out": False,
        "flip": False,
        "gaussian_blur": False,
        "random_crop": False,
        "resize": True,
        "rotate": True,
    }
    train_param.epochs = 50
    train_param.batch_size = 128
    # train_param, model_param = process_experiment_args(args, console_logger)
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
        save_dir=SAVED_META_INFO_PATH,
        experiment_name=prepare_name(f"ssl_{args.checkpoint}_", train_param),
    )

    # model.

    supervised_head_param = edict(read_json(SSL_CONFIG))
    supervised_head_param.num_samples = len(data)
    # experiment 4
    # GB
    # supervised_head_param.saved_model_name = "408c17e7b54543d1b026d947ed1365d7" # GB
    # supervised_head_param.saved_model_name = "b815c7ea22484c3a84f89657d98fe2f5" #RC
    # supervised_head_param.saved_model_name = "2ad8250400df46e3936c3299c9e8f491" #C
    # supervised_head_param.saved_model_name = "a71d437614b74ecb89c763b905b9d07c" #CD

    # experiment 4b
    # C
    # supervised_head_param.saved_model_name ="e95c7234c7cb488585969421f9f11fa6" #

    # rC
    # supervised_head_param.saved_model_name ="0944d7a80c6d4d96aaf3083a5675bb4a" #89,59,94

    # GB
    # supervised_head_param.saved_model_name ="3a8bc4245e234bf7a7679ec67b65f449" #44,74,99

    # CJ
    # supervised_head_param.saved_model_name ="fbb9ee970da5471bba8558b5eb21787c" #84,69,74

    # CD
    supervised_head_param.saved_model_name = (
        "fb43fda44dd0410c9329dedf968f35ca"  # 19,29,24
    )

    supervised_head_param.checkpoint = f"epoch={args.checkpoint}.ckpt"
    model = SupervisedHead(supervised_head_param)

    # callbacks
    logging_interval = "step"
    upload_comet_logs = UploadCometLogs(
        logging_interval, get_console_logger("callback"), "supervised"
    )
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    # Trainer setup

    trainer = Trainer(
        accumulate_grad_batches=1,
        gpus="0",
        logger=comet_logger,
        max_epochs=train_param.epochs,
        precision=16,
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
    trainer.logger.experiment.log_parameters({"model_param": supervised_head_param})
    trainer.fit(model, train_data_loader, val_data_loader)

    # evaluation:
    model.eval()

    data.is_training(False)
    results = evaluate(
        model,
        data,
        num_workers=train_param.num_workers,
        batch_size=train_param.batch_size,
    )
    with trainer.logger.experiment.validate():
        trainer.logger.experiment.log_metrics(results)

    data.is_training(True)
    results = evaluate(
        model,
        data,
        num_workers=train_param.num_workers,
        batch_size=train_param.batch_size,
    )
    with trainer.logger.experiment.train():
        trainer.logger.experiment.log_metrics(results)


if __name__ == "__main__":
    main()
