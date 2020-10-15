import os
from typing import Tuple

import numpy as np
import torchvision
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from src.constants import FREIHAND_DATA
from src.data_loader.freihand_loader import F_DB
from src.data_loader.sample_augmenter import SampleAugmenter
from src.data_loader.utils import convert_2_5D_to_3D, convert_to_2_5D
from torch.utils.data import Dataset


class Data_Set(Dataset):
    def __init__(
        self,
        config: edict,
        transform: torchvision.transforms,
        train_set: bool = True,
        experiment_type: str = "supervised",
    ):
        """This class acts as overarching data_loader.
        It coordinates the indices that must go to the train and validation set.
        Note: To switch between train and validation switch the mode using ``is_training(True)`` for
        training and is_training(False) for validation.
        To create simulatenous instances of validation and training, make a shallow copy and change the
        mode with ``is_training()``

        Args:
            config (e): Configuraction dict must have  "seed" and "train_ratio".
            transforms ([type]): torch transforms or composition of them.
            train_set (bool, optional): Flag denoting which samples are returned. Defaults to True.
            experiment_type (str, optional): Flag denoting how to decide what should be the sample format.
                Default is "supervised". For SimCLR change to "simclr"
        """
        # Individual data loader initialization.
        self.config = config
        self.f_db = F_DB(
            root_dir=os.path.join(FREIHAND_DATA, "training", "rgb"),
            labels_path=os.path.join(FREIHAND_DATA, "training_xyz.json"),
            camera_param_path=os.path.join(FREIHAND_DATA, "training_K.json"),
            config=self.config,
        )
        self.transform = transform
        self.config = config
        self.augmenter = self.get_sample_augmenter()
        self._train_set = train_set
        self.experiment_type = experiment_type
        # The real amount of input images (excluding the augmented background.)
        self._size_f_db = len(self.f_db) // 4
        self.f_db_train_indices, self.f_db_val_indices = self.get_f_db_indices()

    def __getitem__(self, idx: int):
        # As of now all the data would be passed as is,
        #  because there is only one dataset.
        if self._train_set:
            sample = self.f_db[self.f_db_train_indices[idx]]
        else:
            sample = self.f_db[self.f_db_val_indices[idx]]
        if self.experiment_type == "simclr":
            sample = self.prepare_simclr_sample(sample)
        else:
            sample = self.prepare_supervised_sample(sample)
        return sample

    def __len__(self):
        if self._train_set:
            return sum([len(self.f_db_train_indices)])
        else:
            return sum([len(self.f_db_val_indices)])

    def get_sample_augmenter(self):
        return SampleAugmenter(
            augmentation_params=self.config.augmentation_params,
            augmentation_flags=self.config.augmentation_flags,
        )

    def prepare_simclr_sample(self, sample: dict) -> dict:
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        img1, _ = self.augmenter.transform_sample(sample["image"], joints25D)
        img2, _ = self.augmenter.transform_sample(sample["image"], joints25D)
        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return {"transformed_image1": img1, "transformed_image2": img2}

    def prepare_supervised_sample(self, sample: dict) -> dict:
        joints25D, scale = convert_to_2_5D(sample["K"], sample["joints3D"])
        image, joints25D = self.augmenter.transform_sample(sample["image"], joints25D)
        joints3D_recreated = convert_2_5D_to_3D(joints25D, scale, sample["K"])
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "joints": joints25D,
            "joints3D": sample["joints3D"],
            "K": sample["K"],
            "scale": scale,
            "joints_3D_recreated": joints3D_recreated,
        }

    def is_training(self, value: bool):
        """Switches the mode of the data.

        Args:
            value (bool): If value is True then training samples are returned else
        validation samples are returned.
        """
        self._train_set = value

    def get_f_db_indices(self) -> Tuple[np.array, np.array]:
        """Randomly samples the trainnig and validation indices for Freihand dataset.
        Since Freihand data is augmented 4 times by chnaging background the validation set is created only
        from the same real image images and there augmentations.

        Returns:
            Tuple[np.array, np.array]: Tuple of  train and validation indices respectively.
        """
        train_indices, val_indices = train_test_split(
            np.arange(0, self._size_f_db),
            train_size=self.config.train_ratio,
            random_state=self.config.seed,
        )
        train_indices = np.sort(train_indices)
        val_indices = np.sort(val_indices)
        train_indices = np.concatenate(
            (
                train_indices,
                train_indices + self._size_f_db,
                train_indices + self._size_f_db * 2,
                train_indices + self._size_f_db * 3,
            ),
            axis=0,
        )
        val_indices = np.concatenate(
            (
                val_indices,
                val_indices + self._size_f_db,
                val_indices + self._size_f_db * 2,
                val_indices + self._size_f_db * 3,
            ),
            axis=0,
        )
        return train_indices, val_indices
