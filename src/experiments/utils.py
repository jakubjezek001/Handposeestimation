import argparse
import os
from logging import Logger
from pprint import pformat
from typing import List, Tuple

import torch
from comet_ml import Experiment
from easydict import EasyDict as edict
from src.constants import (
    DATA_PATH,
    MASTER_THESIS_DIR,
    SUPERVISED_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    SAVED_MODELS_BASE_PATH,
    SAVED_META_INFO_PATH,
)
from src.data_loader.data_set import Data_Set
from src.models.utils import get_latest_checkpoint
from src.utils import read_json
from src.experiments.evaluation_utils import evaluate


def get_general_args(
    description: str = "Script for training baseline supervised model",
) -> argparse.Namespace:
    """Function to parse the arguments given as input to a general experiment.
    only parses augmentation flag and data parameters like training ratio, num_workers,
    batchsize, epochs.
    Returns:
       argparse.Namespace: Parsed arguments as namespace.
    """

    parser = argparse.ArgumentParser(description=description)

    # Augmentation flags
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
    parser.add_argument(
        "--sobel_filter", action="store_true", help="To enable sobel filtering"
    )
    parser.add_argument(
        "--gaussian_noise", action="store_true", help="To add gaussian noise."
    )
    parser.add_argument("-tag", action="append", help="Tag for comet", default=[])
    # Training  and data loader params.
    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    parser.add_argument("-seed", type=int, help="To add seed")
    parser.add_argument(
        "-num_workers", type=int, help="Number of workers for Dataloader."
    )
    parser.add_argument(
        "-train_ratio", type=float, help="Ratio of train:validation split."
    )
    parser.add_argument(
        "-accumulate_grad_batches",
        type=int,
        help="Number of batches to accumulate gradient.",
    )
    parser.add_argument(
        "--denoiser", action="store_true", help="To enable denoising", default=False
    )
    args = parser.parse_args()
    return args


def get_hybrid1_args(
    description: str = "Script for training hybrid1 model",
) -> argparse.Namespace:
    """Function to parse the arguments given as input to a hybrid1 experiment.
    Returns:
       argparse.Namespace: Parsed arguments as namespace.
    """

    parser = argparse.ArgumentParser(description=description)

    # Augmentation flags
    parser.add_argument(
        "-contrastive",
        action="append",
        help="Add augmentations for contrastive sample.",
        choices=["rotate", "crop", "color_jitter"],
    )
    parser.add_argument(
        "-pairwise",
        action="append",
        help="Add augmentations for pairwise sample.",
        choices=["rotate", "crop", "color_jitter"],
    )
    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-tag", action="append", help="Tag for comet", default=[])
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    parser.add_argument("-seed", type=int, help="To add seed")
    parser.add_argument(
        "-num_workers", type=int, help="Number of workers for Dataloader."
    )
    parser.add_argument(
        "-train_ratio", type=float, help="Ratio of train:validation split."
    )
    parser.add_argument(
        "-accumulate_grad_batches",
        type=int,
        help="Number of batches to accumulate gradient.",
    )
    args = parser.parse_args()
    return args


def update_hybrid1_train_args(args: argparse.Namespace, train_param: edict) -> edict:
    if args.pairwise is not None:
        for item in args.pairwise:
            train_param.pairwise.augmentation_flags[item] = True
    if args.contrastive is not None:
        for item in args.contrastive:
            train_param.contrastive.augmentation_flags[item] = True
    if args.train_ratio is not None:
        train_param.train_ratio = (args.train_ratio * 100 % 100) / 100.0
    if args.train_ratio is not None:
        train_param.train_ratio = (args.train_ratio * 100 % 100) / 100.0
    if args.accumulate_grad_batches is not None:
        train_param.accumulate_grad_batches = args.accumulate_grad_batches
    train_param.update(
        update_param(
            args,
            train_param,
            ["batch_size", "epochs", "train_ratio", "num_workers", "seed"],
        )
    )
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
            "sobel_filter",
            "gaussian_noise",
        ],
    )
    if args.accumulate_grad_batches is not None:
        train_param.accumulate_grad_batches = args.accumulate_grad_batches
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


def prepare_name(prefix: str, train_param: edict, hybrid_naming: bool = False) -> str:
    """Encodes the train paramters into string for appropraite naming of experiment.

    Args:
        prefix (str): prefix to attach to the name example sup , simclr, ssl etc.
        train_param (edict): train params used for the experiment.

    Returns:
        str: name of the experiment.
    """
    codes = {
        "color_drop": "CD",
        "color_jitter": "CJ",
        "crop": "C",
        "cut_out": "CO",
        "flip": "F",
        "gaussian_blur": "GB",
        "random_crop": "RC",
        "resize": "Re",
        "rotate": "Ro",
        "sobel_filter": "SF",
        "gaussian_noise": "GN",
    }
    if hybrid_naming:
        pairwise_augmentations = "_".join(
            sorted(
                [
                    codes[key]
                    for key, value in train_param.pairwise.augmentation_flags.items()
                    if value
                ]
            )
        )
        contrastive_augmentations = "_".join(
            sorted(
                [
                    codes[key]
                    for key, value in train_param.contrastive.augmentation_flags.items()
                    if value
                ]
            )
        )
        return (
            f"{prefix}{train_param.batch_size}_rel_{pairwise_augmentations}"
            f"_con_{contrastive_augmentations}"
        )

    else:
        augmentations = "_".join(
            sorted(
                [
                    codes[key]
                    for key, value in train_param.augmentation_flags.items()
                    if value
                ]
            )
        )

        return f"{prefix}{train_param.batch_size}{augmentations}"


def save_experiment_key(
    experiment_name: str, experiment_key: str, filename="default.csv"
):
    """Writes the experiemtn name and key in a  file for quick reference to use the
    saved models.

    Args:
        experiment_name (str]): Name of the experiment. from prepare_name()
        experiment_key (str): comet generated experiment key.
        filename (str, optional): Name of the file where the info should be appended.
         Defaults to "default.csv".
    """
    with open(os.path.join(SAVED_META_INFO_PATH, filename), "a") as f:
        f.write(f"{experiment_name},{experiment_key}\n")


def get_nips_a1_args():
    parser = argparse.ArgumentParser(
        description="Experiment NIPS A1: SIMCLR ablative studies"
    )
    parser.add_argument(
        "augmentation", type=str, default=None, help="Select augmentation to apply"
    )
    args = parser.parse_args()
    return args


def get_nips_a2_args():
    parser = argparse.ArgumentParser(
        description="Experiment NIPS A2: Pairwise ablative studies"
    )
    parser.add_argument(
        "augmentation", type=str, default=None, help="Select augmentation to apply"
    )
    args = parser.parse_args()
    return args


def get_downstream_args():
    parser = argparse.ArgumentParser(description="Downstream training experiment")
    parser.add_argument("experiment_key", type=str, default=None, help="Experiment key")
    parser.add_argument(
        "experiment_name",
        type=str,
        default=None,
        help="Name of the pretrained experiment",
    )
    parser.add_argument(
        "experiment_type",
        type=str,
        default=None,
        help="Type of experiment for tagging.",
    )
    args = parser.parse_args()
    args = parser.parse_args()
    return args


def downstream_evaluation(
    model, data: Data_Set, num_workers: int, batch_size: int, logger: Experiment
) -> Tuple[dict, dict]:
    """Returns train and validate results respectively.

    Args:
        model ([type]): [description]
        data (Data_Set): [description]
        num_workers (int): [description]
        batch_size (int): [description]
        logger (Experiment):

    Returns:
        Tuple[dict, dict]: [description]
    """

    model.eval()
    data.is_training(False)
    validate_results = evaluate(
        model, data, num_workers=num_workers, batch_size=batch_size
    )
    with logger.experiment.validate():
        logger.experiment.log_metrics(validate_results)

    # data.is_training(True)
    # train_results = evaluate(
    #     model, data, num_workers=num_workers, batch_size=batch_size
    # )

    # with logger.experiment.train():
    #     logger.experiment.log_metrics(train_results)


def restore_model(model, experiment_key: str, checkpoint: str = ""):
    """Restores the experiment with the most recent checkpoint.

    Args:
        experiment_key (str): experiment key
    """
    saved_state_dict = torch.load(get_latest_checkpoint(experiment_key, checkpoint))[
        "state_dict"
    ]
    model.load_state_dict(saved_state_dict)
    return model


def get_checkpoints(experiment_key: str, number: int = 3) -> List[str]:
    """Returns last 'n' checkpoints.

    Args:
        experiment_key (str): [description]
        number (int, optional): [description]. Defaults to 3.

    Returns:
        List[str]: Name of last n checkpoints.
    """
    return sorted(
        os.listdir(os.path.join(SAVED_MODELS_BASE_PATH, experiment_key, "checkpoints"))
    )[::-1][:number]
