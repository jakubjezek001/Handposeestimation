import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data_loader.utils import convert_to_2_5D


class F_DB(Dataset):
    """Class to load samples from the Fre hand dataset.
    Inherits from the Dataset class in  torch.utils.data
    """

    def __init__(
        self, root_dir: str, labels_path: str, camera_param_path: str, transform
    ):
        self.root_dir = root_dir
        self.labels = self.get_labels(labels_path)
        self.camera_param = self.get_camera_param(camera_param_path)
        self.img_names = self.get_image_names()
        self.transform = transform

    def get_image_names(self):
        img_names = next(os.walk(self.root_dir))[2]
        img_names.sort()
        return img_names

    def get_labels(self, lables_path):
        with open(lables_path, "r") as f:
            return json.load(f)

    def get_camera_param(self, camera_param_path):
        with open(camera_param_path, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_names[idx])
        img = Image.open(img_name)
        joints3D = torch.tensor(self.labels[idx]).float()
        camera_param = torch.tensor(self.camera_param[idx]).float()
        joints25D, scale = convert_to_2_5D(camera_param, joints3D)

        sample = {
            "image": img,
            "joints": joints25D,
            "scale": scale,
            "K": camera_param,
            "joints_3D": joints3D,
        }
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample
