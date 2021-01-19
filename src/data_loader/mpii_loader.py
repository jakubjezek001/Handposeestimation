import os
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
from src.data_loader.joints import Joints
from src.utils import read_json
from torch.utils.data import Dataset


class MPII_DB(Dataset):
    def __init__(self, root_dir: str, split: str = "train"):
        """Initializes the MPII dataset class, relevant paths and the Joints
        initializes the class for remapping of MPII formatted joints to that of AIT.
        joints mapping at
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#hand-output-format

        Args:
            root_dir (str): Path to the directory with image samples.
            split (str): To select train or test split.
        """
        self.root_dir = root_dir
        self.split = split
        self.image_dir_path = os.path.join(self.root_dir, f"manual_{self.split}")
        self.label_dir_path = os.path.join(self.root_dir, f"manual_{self.split}")
        self.img_names = self.get_image_names()
        self.labels = self.get_labels()

        # To convert from MPII to AIT format.
        self.joints = Joints()

    def get_image_names(self) -> List[str]:
        """Gets the name of all the files in root_dir.
        Make sure there are only image in that directory as it reads all the file names.

        Returns:
            List[str]: List of image names.
        """

        img_names = [
            file_name
            for file_name in next(os.walk(self.image_dir_path))[2]
            if ".jpg" in file_name
        ]
        img_names.sort()
        # popping, images with annottaaions out of image bounds
        img_names.remove("Ricki_unit_8.flv_000003_l.jpg")
        img_names.remove("Ricki_unit_8.flv_000002_l.jpg")
        return img_names

    def get_labels(self) -> Dict[str, dict]:
        label_file_names = [
            file_name
            for file_name in next(os.walk(self.label_dir_path))[2]
            if ".json" in file_name
        ]
        labels = {
            file_name.replace(".json", ""): read_json(
                os.path.join(self.label_dir_path, file_name)
            )["hand_pts"]
            for file_name in label_file_names
        }
        return labels

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.array, torch.Tensor]]:
        """Returns a sample corresponding to the index.

        Args:
            idx (int): index

        Returns:
            dict: item with following elements.
                "image" in opencv bgr format.
                "K": camera params (Indetity matrix in this case)
                "joints3D": 3D coordinates of joints in AIT format. (z coordinate is 1.0)
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir_path, self.img_names[idx])
        img = cv2.imread(img_name)
        # mpii follow the same strategy as the freihand for joint naming.
        joints3D = self.joints.freihand_to_ait(
            torch.tensor(self.labels[self.img_names[idx].replace(".jpg", "")]).float()
        )
        camera_param = torch.eye(3).float()
        joints_valid = torch.ones_like(joints3D[..., -1])
        sample = {
            "image": img,
            "K": camera_param,
            "joints3D": joints3D,
            "joints_valid": torch.tensor(joints_valid),
        }

        return sample
