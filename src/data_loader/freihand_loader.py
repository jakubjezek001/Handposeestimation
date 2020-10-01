import os
import random
from typing import List, Tuple

import torch
from PIL import Image
from src.constants import ANGLES
from src.data_loader.joints import Joints
from src.data_loader.utils import (
    convert_to_2_5D,
    sample_cropper,
    sample_rotator,
    sample_resizer,
)
from src.types import JOINTS_25D
from src.utils import read_json
from torch.utils.data import Dataset
from torchvision import transforms

CROP_MARGIN = 1.5


class F_DB(Dataset):
    """Class to load samples from the Freihand dataset.
    Inherits from the Dataset class in  torch.utils.data.
    Note: The keypoints are mapped to format used at AIT.
    Refer to joint_mapping.json in src/data_loader/utils.
    """

    def __init__(
        self, root_dir: str, labels_path: str, camera_param_path: str, transform, config
    ):
        """Initializes the freihand dataset class, relevant paths and the Joints
        class for remapping of freihand formatted joints to that of AIT.

        Args:
            root_dir (str): Path to the directory with image samples.
            labels_path (str): Path to the training labels json.
            camera_param_path (str): Path to the camera param json
            transform ([type]): Transforms that needs to be applied to the image.
        """
        self.root_dir = root_dir
        self.labels = self.get_labels(labels_path)
        self.camera_param = self.get_camera_param(camera_param_path)
        self.img_names = self.get_image_names()
        self.transform = transform
        # To convert from freihand to AIT format.
        self.joints = Joints()
        self.config = config
        random.seed(self.config.seed)  # To control randomness in rotation

    def get_image_names(self) -> List[str]:
        """Gets the name of all the files in root_dir.
        Make sure there are only image in that directory as it reads all the file names.

        Returns:
            List[str]: List of image names.
        """
        img_names = next(os.walk(self.root_dir))[2]
        img_names.sort()
        return img_names

    def get_labels(self, lables_path: str) -> list:
        """Extacts the labels(joints coordinates) from the label_json at labels_path

        Args:
            lables_path (str): Path to labels json.

        Returns:
            list: List of all the the coordinates(32650).
        """
        return read_json(lables_path)

    def get_camera_param(self, camera_param_path: str) -> list:
        """Extacts the camera parameters from the camera_param_json at camera_param_path.

        Args:
            camera_param_path (str): Path to json containing camera paramters.

        Returns:
            list: List of camera paramters for all images(32650)
        """
        return read_json(camera_param_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # TODO: Write appropriate docstring once the output is finalized.
        # Remove unnecessary data that is being passed.

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.img_names[idx])
        img = Image.open(img_name)
        joints3D = self.joints.freihand_to_ait(
            torch.tensor(self.labels[idx % 32560]).float()
        )
        camera_param = torch.tensor(self.camera_param[idx % 32560]).float()
        joints25D, scale = convert_to_2_5D(camera_param, joints3D)
        # Applying sample related transforms
        img, joints25D = self.apply_transforms(img, joints25D)

        sample = {
            "image": img,
            "joints": joints25D,
            "scale": scale,
            "K": camera_param,
            "joints_3D": joints3D,
        }
        # Applying only image realted transform
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

    def apply_transforms(
        self, image: Image.Image, joints: JOINTS_25D
    ) -> Tuple[Image.Image, JOINTS_25D]:
        if self.config.rotate:
            angle = random.choice(ANGLES)
            image, joints = sample_rotator(image, joints, angle)
        if self.config.crop:
            image, joints = sample_cropper(
                image, joints, self.config.crop_margin, self.config.crop_keypoints
            )
        if self.config.resize:
            image, joints = sample_resizer(
                image, joints, self.config.resize_shape, self.config.resize_keypoints
            )
        return image, joints
