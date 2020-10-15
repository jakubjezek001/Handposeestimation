import argparse
from logging import Logger
from pprint import pformat
import copy
from typing import List, Tuple

from easydict import EasyDict as edict
from src.constants import MODEL_CONFIG_PATH, TRAINING_CONFIG_PATH
from src.utils import read_json
from src.data_loader.data_set import Data_Set
from torch.utils.data import DataLoader


def get_experiement_args() -> argparse.Namespace:
    """Function to parse the arguments given as input to the experiment scripts.

    Returns:
       argparse.Namespace: Parsed arguments as namespace.
    """
    parser = argparse.ArgumentParser(description="Script for training a model")
    parser.add_argument(
        "--cpu", action="store_true", help="Select this option to use CPU training"
    )
    parser.add_argument(
        "-resnet_trainable",
        help="True for trainable and false for not trainable. Defaut according to config.",
    )
    parser.add_argument(
        "-train_ratio", type=float, help="Ratio of train:validation split."
    )
    parser.add_argument(
        "-crop", type=bool, help="To crop the image around hand coordinates,"
    )
    parser.add_argument(
        "-crop_keypoints", type=bool, help="To crop the joints IF image is cropped,"
    )
    parser.add_argument(
        "-crop_margin",
        type=float,
        help="To enlarge the crop box, values will be clipped between 1 and 2",
    )
    parser.add_argument("--rotate", type=bool, help="To rotate samples randomly")
    parser.add_argument("-learning_rate", type=float, help="Learning _rate.")
    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    parser.add_argument("-seed", type=int, help="To add random seed")
    parser.add_argument(
        "-num_workers", type=int, help="Number of workers for Dataloader."
    )
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
    train_param = update_train_params(args, train_param)
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
    train_param = update_param(
        args,
        train_param,
        [
            "batch_size",
            "epochs",
            "train_ratio",
            "num_workers",
            "seed",
            "rotate",
            "crop",
            "crop_margin",
            "crop_keypoints",
            "resnet_trainable",
        ],
    )
    train_param.gpu = True if args.cpu is not True else False
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
    model_param = update_param(args, model_param, ["resnet_trainable", "learning_rate"])
    model_param.gpu = True if args.cpu is not True else False
    return model_param


def get_train_val_split(data: Data_Set, **kwargs) -> Tuple[DataLoader, DataLoader]:
    data.is_training(True)
    val_data = copy.copy(data)
    val_data.is_training(False)
    return DataLoader(data, **kwargs), DataLoader(val_data, **kwargs)
