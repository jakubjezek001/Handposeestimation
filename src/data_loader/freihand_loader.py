import os
from typing import List

import numpy as np
import torch
from PIL import Image
from src.data_loader.joints import Joints
from src.data_loader.utils import convert_to_2_5D
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
        self, root_dir: str, labels_path: str, camera_param_path: str, transform
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
        # To convert freihad to AIT format.
        self.joints = Joints()

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

        sample = {
            "image": img,
            "joints": joints25D,
            "scale": scale,
            "K": camera_param,
            "joints_3D": joints3D,
        }
        sample["image"] = self.image_cropper(sample["image"], joints25D, CROP_MARGIN)

        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

    def image_cropper(
        self, image: Image, joints: JOINTS_25D, crop_margin: float
    ) -> Image:
        """Uses the image coordinates to extract the location of the hand and then cropping around the handpose.

        Args:
            image (PIL.Image): A PIL image.
            joints (JOINTS_25D): tensor of all 21 keypoints
            crop_margin (float): The amount by which the crop box should be scaled. valid range 1 to 2. Other values will be clipped

        Returns:
            Image:  A cropped PIL image.
        """
        crop_margin = np.clip(crop_margin, 1.0, 2.0)
        top, left = torch.min(joints[:, 1]), torch.min(joints[:, 0])
        bottom, right = torch.max(joints[:, 1]), torch.max(joints[:, 0])
        height, width = bottom - top, right - left
        return transforms.functional.crop(
            image,
            top=int(top - height * (crop_margin - 1) / 2),
            left=int(left - width * (crop_margin - 1) / 2),
            height=int(height * crop_margin),
            width=int(width * crop_margin),
        )
