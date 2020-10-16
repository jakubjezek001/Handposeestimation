import argparse
from logging import Logger
from pprint import pformat
from typing import List

from easydict import EasyDict as edict
from src.constants import MODEL_CONFIG_PATH, TRAINING_CONFIG_PATH
from src.utils import read_json


def get_experiement_args() -> argparse.Namespace:
    """Function to parse the arguments given as input to the experiment scripts.

    Returns:
       argparse.Namespace: Parsed arguments as namespace.
    """
    parser = argparse.ArgumentParser(description="Script for training a model")
    parser.add_argument("--cpu", action="store_true", help="Eanbles CPU training")
    parser.add_argument("-lr", type=float, help="Learning _rate.")
    parser.add_argument("-opt_weight_decay", type=int, help="Weight decay")
    parser.add_argument("-warmup_epochs", type=int, help="Number of warmup epochs")
    parser.add_argument("-precision", type=int, choices=[16, 32], default=16)

    # Augmenter flags
    parser.add_argument(
        "--color_drop", action="store_true", help="To enable random color drop"
    )
    parser.add_argument(
        "--color_jitter", action="store_true", help="To enable random jitter"
    )
    parser.add_argument("--crop", action="store_true", help="To enable cropping")
    parser.add_argument(
        "--cut_out", action="store_true", help="To enable random cur out"
    )
    parser.add_argument("--flip", action="store_true", help="To enable random flipping")
    parser.add_argument(
        "--gaussian_blur", action="store_true", help="To enable gaussina blur"
    )
    parser.add_argument(
        "--rotate", action="store_true", help="To rotate samples randomly"
    )
    parser.add_argument(
        "--random_crop", action="store_true", help="To enable random cropping"
    )
    parser.add_argument("--resize", action="store_true", help="To enable resizing")

    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    parser.add_argument("-seed", type=int, help="To add random seed")
    parser.add_argument(
        "-num_workers", type=int, help="Number of workers for Dataloader."
    )
    parser.add_argument(
        "-train_ratio", type=float, help="Ratio of train:validation split."
    )
    parser.add_argument(
        "-resnet_trainable",
        help="True for trainable and false for not trainable. Defaut according to config.",
    )

    parser.add_argument("-crop_margin", type=float, help="Change the crop margin.")

    args = parser.parse_args()
    return args


def process_experiment_args(args: argparse.Namespace, console_logger: Logger) -> edict:
    """Reads the training parameters and adjusts them according to
        the arguments passed to the experiment script.

    Args:
        args (argparse.Namespace): Arguments from get_experiement_args().
        console_logger (Logger): logger object

    Returns:
        edict: Updated training parameters.
    """
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    model_param = edict(read_json(MODEL_CONFIG_PATH))

    args = get_experiement_args()
    # console_logger.info(f"Default config ! {pformat(train_param)}")
    train_param.accumulate_grad_batches = 1
    train_param = update_train_params(args, train_param)
    if train_param.batch_size > 256:
        train_param.accumulate_grad_batches = int(train_param.batch_size // 256)
        train_param.batch_size = 256
        model_param.batch_size = 256

    console_logger.info(f"Training configurations {pformat(train_param)}")
    # console_logger.info(f"Default Model config ! {pformat(model_param)}")
    model_param = update_model_params(args, model_param)
    console_logger.info(f"Model configurations {pformat(model_param)}")
    return train_param, model_param


def update_train_params(args: argparse.Namespace, train_param: edict) -> edict:
    """Updates and returns the training hyper paramters as per args

    Args:
        args (argparse.Namespace): Arguments from get_experiement_args().
        train_param (edict): Default training parameter.

    Returns:
        edict: Updated training parameters.
    """
    if args.train_ratio is not None:
        train_param.data.train_ratio = (args.train_ratio * 100 % 100) / 100.0
    train_param.update(
        update_param(
            args,
            train_param,
            ["batch_size", "epochs", "train_ratio", "num_workers", "seed"],
        )
    )
    train_param.augmentation_flags = update_param(
        args,
        train_param.augmentation_flags,
        [
            "color_drop",
            "color_jitter",
            "crop",
            "cut_out",
            "flip",
            "gaussian_blur",
            "random_crop",
            "resize",
            "rotate",
        ],
    )
    train_param.gpu = True if args.cpu is not True else False
    train_param.precision = args.precision
    return train_param


def update_param(args: argparse.Namespace, config: edict, params: List[str]) -> edict:
    """Update the config according to the argument.

    Args:
        args (edict): script arguments
        config (edict): configuration as read from json
        params (List[str]): Name of paramters that must be edited.

    Returns:
        edict: Updated config.
    """
    args_dict = vars(args)
    for param in params:
        if args_dict[param] is not None:
            config[param] = args_dict[param]
    return config


def update_model_params(args: argparse.Namespace, model_param: edict) -> edict:
    """Updates the model parameters according to experiment args

    Args:
        args (argparse.Namespace):Arguments from get_experiement_args().
        model_param (edict): [description]

    Returns:
        edict:Updated model parameters.
    """
    model_param = update_param(
        args,
        model_param,
        ["resnet_trainable", "lr", "batch_size", "opt_weight_decay", "warmup_epochs"],
    )
    model_param.gpu = True if args.cpu is not True else False
    return model_param
