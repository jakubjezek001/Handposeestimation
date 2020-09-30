import argparse

from easydict import EasyDict as edict
from src.constants import MODEL_CONFIG_PATH, TRAINING_CONFIG_PATH
from src.utils import read_json


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
    parser.add_argument("-learning_rate", type=float, help="Learning _rate.")
    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "-num_workers", type=int, help="Number of workers for Dataloader."
    )
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
    model_param = edict(read_json(MODEL_CONFIG_PATH))
    args = get_experiement_args()
    print(f"Default config ! {train_param}")
    train_param = update_train_params(args, train_param)
    print(f"Updated configurations {train_param}")
    print(f"Default Model config ! {model_param}")
    model_param = update_model_params(args, model_param)
    print(f"Updated configurations {model_param}")
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
    if args.batch_size is not None:
        train_param.batch_size = args.batch_size
    if args.epochs is not None:
        train_param.epochs = args.epochs
    if args.num_workers is not None:
        train_param.num_workers = args.num_workers

    train_param.resnet_trainable = args.resnet_trainable
    train_param.gpu = True if args.cpu is not True else False
    return train_param


def update_model_params(args: argparse.Namespace, model_param: edict) -> edict:
    """Updates the model parameters according to experiment args

    Args:
        args (argparse.Namespace):Arguments from get_experiement_args().
        model_param (edict): [description]

    Returns:
        edict:Updated model parameters.
    """
    if args.resnet_trainable is not None:
        model_param.resnet_trainable = args.resnet_trainable
    if args.learning_rate is not None:
        model_param.learning_rate = args.learning_rate
    model_param.gpu = True if args.cpu is not True else False
    return model_param
