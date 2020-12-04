import os
import random
from types import prepare_class
from typing import Tuple

import numpy as np
import torch
import torchvision
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from torchvision import transforms
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

        See 03-Data_handler.ipynb for visualization.
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
        self.experiment_type = experiment_type
        self._train_set = train_set

        if self.experiment_type == "hybrid":
            # Two augmenters are used when hybrid experiment data params are passed.
            self.pairwise_augmenter = self.get_sample_augmenter(
                config.augmentation_params, config.pairwise.augmentation_flags
            )
            self.contrastive_augmenter = self.get_sample_augmenter(
                config.augmentation_params, config.contrastive.augmentation_flags
            )
        else:
            self.augmenter = self.get_sample_augmenter(
                config.augmentation_params, config.augmentation_flags
            )

        # The real amount of input images (excluding the augmented background.)
        self._size_f_db = len(self.f_db) // 4
        self.f_db_train_indices, self.f_db_val_indices = self.get_f_db_indices()

    def __getitem__(self, idx: int):
        # As of now all the data would be passed as is,
        #  because there is only Freihand dataset support.
        if self._train_set:
            sample = self.f_db[self.f_db_train_indices[idx]]
        else:
            sample = self.f_db[self.f_db_val_indices[idx]]

        # Returning data as per the experiment.
        if self.experiment_type == "simclr":
            sample = self.prepare_simclr_sample(sample, self.augmenter)
        elif self.experiment_type == "pairwise":
            sample = self.prepare_pairwise_sample(sample, self.augmenter)
        elif self.experiment_type == "experiment4_pretraining":
            # for simclr ablative, for nips A1
            sample = self.prepare_experiment4_pretraining(sample, self.augmenter)
        elif self.experiment_type == "pairwise_ablative":
            # for nips A2
            sample = self.prepare_pairwise_ablative(sample, self.augmenter)
        elif self.experiment_type == "hybrid":
            sample = self.prepare_hybrid1_sample(
                sample, self.contrastive_augmenter, self.pairwise_augmenter
            )
        else:
            sample = self.prepare_supervised_sample(sample, self.augmenter)
        return sample

    def __len__(self):
        if self._train_set:
            return sum([len(self.f_db_train_indices)])
        else:
            return sum([len(self.f_db_val_indices)])

    def get_sample_augmenter(
        self, augmentation_params: edict, augmentation_flags: edict
    ) -> SampleAugmenter:
        return SampleAugmenter(
            augmentation_params=augmentation_params,
            augmentation_flags=augmentation_flags,
        )

    def prepare_simclr_sample(self, sample: dict, augmenter: SampleAugmenter) -> dict:
        """Prepares sample according to SimCLR experiment.
        For each sample two transformations of an image are returned.
        Note: Rotation and jitter is kept same in both the transformations.
        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing 'transformed_image1' and 'transformed_image2'
        """
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        img1, _, _ = augmenter.transform_sample(sample["image"], joints25D.clone())

        # To keep rotation and jitter consistent between the two transformations.
        override_angle = augmenter.angle
        overrride_jitter = augmenter.jitter

        img2, _, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, overrride_jitter
        )

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return {"transformed_image1": img1, "transformed_image2": img2}

    def prepare_experiment4_pretraining(
        self, sample: dict, augmenter: SampleAugmenter
    ) -> dict:
        """Prepares samples for ablative studies on Simclr. This function isolates the
        effect of each transform. Make sure no other transformation is applied except
        the one you want to isolate. (Resize is allowed). Samples are not
        artificially increased by changing rotation and jitter for both samples.

        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing 'transformed_image1' and 'transformed_image2'
        """

        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        if augmenter.crop:
            override_jitter = None
        else:
            # Zero jitter is added incase the cropping is off. It is done to trigger the
            # cropping but always with no translation in image.
            override_jitter = [0, 0]
        if augmenter.rotate:
            override_angle = None
        else:
            override_angle = None
            # override_angle = random.uniform(1, 360)
            # uncomment line baove to add this rotation  to both channels

        img1, _, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )
        img2, _, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {"transformed_image1": img1, "transformed_image2": img2}

    def prepare_pairwise_sample(self, sample: dict, augmenter: SampleAugmenter) -> dict:
        """Prepares samples according to pairwise experiment, i.e. transforming the
        image and keepinf track of the relative parameters.
        Note: Gaussian blur and Flip are treated as boolean. Also it was decided not to
        use them for experiment.
        The effects of transformations are isolated.

        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing following elements
                'transformed_image1'
                'transformed_image2'
                'joints1' (2.5D joints)
                'joints2' (2.5D joints)
                'rotation'
                'jitter' ...
        """
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])

        img1, joints1, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone()
        )
        param1 = self.get_random_augment_param(augmenter)

        img2, joints2, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone()
        )
        param2 = self.get_random_augment_param(augmenter)

        # relative transform calculation.
        rel_param = self.get_relative_param(augmenter, param1, param2)

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            **{
                "transformed_image1": img1,
                "transformed_image2": img2,
                "joints1": joints1,
                "joints2": joints2,
            },
            **rel_param,
        }

    def prepare_pairwise_ablative(
        self, sample: dict, augmenter: SampleAugmenter
    ) -> dict:
        """Prepares samples according to pairwise experiment, i.e. transforming the
        image and keeping track of the relative parameters. Augmentations are isolated.
        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing following elements
                'transformed_image1'
                'transformed_image2'
                'joints1' (2.5D joints)
                'joints2' (2.5D joints)
                'rotation'
                'jitter' ...
        """
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        if augmenter.crop:
            override_jitter = None
        else:
            # Zero jitter is added incase the cropping is off. It is done to trigger the
            # cropping but always with no translation in image.
            override_jitter = [0, 0]
        if augmenter.rotate:
            override_angle = None
        else:
            override_angle = None
            # override_angle = random.uniform(1, 360)
            # uncomment line above to add this rotation  to both channels
        img1, joints1, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )
        param1 = self.get_random_augment_param(augmenter)

        img2, joints2, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )
        param2 = self.get_random_augment_param(augmenter)

        # relative transform calculation.
        rel_param = self.get_relative_param(augmenter, param1, param2)

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            **{
                "transformed_image1": img1,
                "transformed_image2": img2,
                "joints1": joints1,
                "joints2": joints2,
            },
            **rel_param,
        }

    def prepare_supervised_sample(
        self, sample: dict, augmenter: SampleAugmenter
    ) -> dict:
        """Prepares samples for supervised experiment with keypoints.

        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing following elements
                'image'
                'joints'
                'joints3D'
                'K'
                'scale'
                'joints3D_recreated'
        """
        joints25D_raw, scale = convert_to_2_5D(sample["K"], sample["joints3D"])
        image, joints25D, transformation_matrix = augmenter.transform_sample(
            sample["image"], joints25D_raw
        )
        # transformation_matrix = torch.inverse(joints25D[1]) @ joints25D_raw[1]
        sample["K"] = torch.Tensor(transformation_matrix) @ sample["K"]
        joints3D_recreated = convert_2_5D_to_3D(joints25D, scale, sample["K"])
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "joints": joints25D,
            "joints3D": sample["joints3D"],
            "K": sample["K"],
            "scale": scale,
            "joints3D_recreated": joints3D_recreated,
        }

    def prepare_hybrid1_sample(
        self,
        sample: dict,
        pairwise_augmenter: SampleAugmenter,
        contrastive_augmenter: SampleAugmenter,
    ) -> dict:
        """Prepares samples for basic Hybrid model

        Args:
            sample (dict): Underlying data from dataloader class.
            pairwise_augmenter (SampleAugmenter): Augmenter used to transform sample for
                Pairwise model
            contrastive_augmenter (SampleAugmenter): Augmenter used to transform sample
                for contrastive model.

        Returns:
            dict : sample_containing
                    contrastive_sample.
                    pariwise_sample.
        """
        pairwise_sample = self.prepare_pairwise_ablative(sample, pairwise_augmenter)
        contrastive_sample = self.prepare_experiment4_pretraining(
            sample, contrastive_augmenter
        )
        return {"contrastive": contrastive_sample, "pairwise": pairwise_sample}

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

    def get_random_augment_param(self, augmenter: SampleAugmenter) -> dict:
        """Reads the random paramters from the augmenter for calulation of relative
        transformation
        Args:
            augmenter (SampleAugmenter): Augmenter used to transform the sample.
        Returns:
            dict: Containsangle
                    'jitter_x' (translation of centriod of hand)
                    'jitter_y' (translation of centriod of hand)
                    'h' (hue factor)
                    's' (sat factor)
                    'a' (brightness factor)
                    'b' (brightness additive term)
                    'blur_flag'
        """
        angle = augmenter.angle
        jitter_x = augmenter.jitter_x
        jitter_y = augmenter.jitter_y
        h = augmenter.h
        s = augmenter.s
        a = augmenter.a
        b = augmenter.b
        blur_flag = augmenter._gaussian_blur
        return {
            "angle": angle,
            "jitter_x": jitter_x,
            "jitter_y": jitter_y,
            "h": h,
            "s": s,
            "a": a,
            "b": b,
            "blur_flag": blur_flag,
        }

    def get_relative_param(self, augmenter, param1: dict, param2: dict) -> dict:
        """Calculates relative parameters between two set of augmentation params.

        Args:
            augmenter (SampleAugmenter): Augmenter used to transform sample
            param1 (dict): 1st image  augmetation parameters
                            (from get_random_augment_param())
            param2 (dict): 2nd image augmentation parameters

        Returns:
            dict: relative transformation param
        """
        rel_param = {}

        if augmenter.crop:
            jitter_x = param1["jitter_x"] - param2["jitter_x"]
            jitter_y = param1["jitter_y"] - param2["jitter_y"]
            rel_param.update({"jitter": torch.tensor([jitter_x, jitter_y])})

        if augmenter.color_jitter:
            h = param1["h"] - param2["h"]
            s = param1["s"] - param2["s"]
            a = param1["a"] - param2["a"]
            b = param1["b"] - param2["b"]
            rel_param.update({"color_jitter": torch.tensor([h, s, a, b])})

        if augmenter.gaussian_blur:
            blur_flag = param1["blur_flag"] ^ param2["blur_flag"]
            rel_param.update({"blur": torch.Tensor([blur_flag * 1])})

        if augmenter.rotate:
            angle = (param1["angle"] - param2["angle"]) % 360
            rel_param.update({"rotation": torch.Tensor([angle])})
        return rel_param
