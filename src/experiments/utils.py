import argparse

from easydict import EasyDict as edict
from src.constants import TRAINING_CONFIG_PATH
from src.utils import read_json


def get_experiement_args() -> argparse.Namespace:
    """Function to parse the arguments given as input to the experiment scripts.

    Returns:
       argparse.Namespace: Parsed arguments as namespace.
    """
    parser = argparse.ArgumentParser(description="Script for training a model")
    parser.add_argument(
        "--gpu", action="store_true", help="Select this option to use GPU training"
    )
    parser.add_argument(
        "--resnet_trainable",
        action="store_true",
        help="Makes underlying resent trainable",
    )
    parser.add_argument(
        "-train_ratio", type=float, help="Ratio of train:validation split."
    )
    parser.add_argument("-learning_rate", type=float, help="Learning _rate.")
    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    args = parser.parse_args()
    return args


def process_experiment_args(args: argparse.Namespace) -> edict:
    """Reads the training parameters and adjusts them according to
        the arguments passed to the experiment script.

    Args:
        args (argparse.Namespace): Arguments from get_experiement_args().

    Returns:
        edict: Updated training parameters.
    """
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    args = get_experiement_args()
    print(f"Default config ! {train_param}")
    train_param = update_train_params(args, train_param)
    print(f"Updated configurations {train_param}")
    return train_param


def update_train_params(args: argparse.Namespace, train_param: edict) -> edict:
    """Updates and returns the training hyper paramters as per args

    Args:
        args (argparse.Namespace): Arguments from get_experiement_args().
        train_param (edict): Default training parameter.

    Returns:
        edict: Updated training parameters.
    """
    if args.train_ratio is not None:
        train_param.train_ratio = (args.train_ratio * 100 % 100) / 100.0
    if args.learning_rate is not None:
        train_param.learning_rate = args.learning_rate
    if args.batch_size is not None:
        train_param.batch_size = args.batch_size
    if args.epochs is not None:
        train_param.epochs = args.epochs
    train_param.resnet_trainable = args.resnet_trainable
    train_param.gpu = args.gpu
    return train_param
