import os
from typing import Tuple

import numpy as np
import torchvision
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from src.constants import FREIHAND_DATA
from src.data_loader.freihand_loader import F_DB
from torch.utils.data import Dataset


class Data_Set(Dataset):
    def __init__(
        self, config: edict, transforms: torchvision.transforms, train_set: bool = True
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
        """
        # Individual data loader initialization.
        self.config = config
        self.f_db = F_DB(
            root_dir=os.path.join(FREIHAND_DATA, "training", "rgb"),
            labels_path=os.path.join(FREIHAND_DATA, "training_xyz.json"),
            camera_param_path=os.path.join(FREIHAND_DATA, "training_K.json"),
            transform=transforms,
            config=self.config,
        )
        self.config = config
        self._train_set = train_set
        # The real amount of input images (excluding the augmented background.)
        self._size_f_db = len(self.f_db) // 4
        self.f_db_train_indices, self.f_db_val_indices = self.get_f_db_indices()

    def __getitem__(self, idx: int):
        # As of now all the data would be passed as is,
        #  because there is only one dataset.
        if self._train_set:
            return self.f_db[self.f_db_train_indices[idx]]
        else:
            return self.f_db[self.f_db_val_indices[idx]]

    def __len__(self):
        if self._train_set:
            return sum([len(self.f_db_train_indices)])
        else:
            return sum([len(self.f_db_val_indices)])

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
