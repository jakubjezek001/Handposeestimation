import os

import numpy as np

# import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.core.lightning import LightningModule
# from PIL import Image
from skimage import io  # , transform

# import seaborn as sns
import torch
import json

# import torchvision
# from torchvision import transforms
from torch.utils.data import Dataset  # , DataLoader

# from torch.nn import functional as F


class F_DB(Dataset):
    def __init__(self, root_dir: str, labels_path: str, gray: bool, transform):
        self.root_dir = root_dir
        self.labels = self.get_labels(labels_path)
        self.img_names = self.get_image_names()
        self.transform = transform
        self.gray = gray

    def get_image_names(self):
        img_names = next(os.walk(self.root_dir))[2]
        img_names.sort()
        return img_names

    def get_labels(self, lables_path):
        with open(lables_path, "r") as f:
            return json.load(f)
        return None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_names[idx])
        img = io.imread(img_name, as_gray=self.gray)
        joints = torch.from_numpy(np.array(self.labels[idx])).float()

        sample = {"image": img, "joints": joints}
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
