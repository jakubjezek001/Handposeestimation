import os
from os.path import join
from typing import List, Tuple

import cv2
import torch
from torch.utils import data
from src.data_loader.utils import sudo_joint_bound
from src.utils import read_json, save_json
from torch.utils.data import Dataset
from tqdm import tqdm


class YTB_DB(Dataset):
    """Class to load samples from the youtube dataset.
    Inherits from the Dataset class in  torch.utils.data.
    Note: The joints returned by this class only signify the 2D bound box.
    Not be used for supervised learning!!
    Camera matrix is unity to fit with the sample augmenter.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = root_dir
        self.split = split
        self.bbox, self.img_list = self.get_bbox_labels_and_images()
        self.img_dict = {item["id"]: item for item in self.img_list}

    def get_bbox_labels_and_images(self) -> Tuple[dict, dict]:
        """Returns the dictionary conatinign the bound box of the image and dictionary
        containig image information.

        Returns:
            Tuple[dict, dict]: bbox, image_dict
                image_dict
                    - `name` - Image name in the form
                        of `youtube/VIDEO_ID/video/frames/FRAME_ID.png`.
                    - `width` - Width of the image.
                    - `height` - Height of the image.
                    - `id` - Image ID.
                bbox
                    - `joints` - 21 joints, containing bound box limits as vertices.
                    - `is_left` - Binary value indicating a right/left hand side.
                    - `image_id` - ID to the corresponding entry in `images`.
                    - `id` - Annotation ID (an image can contain multiple hands).
        """
        data_json_path = os.path.join(self.root_dir, f"youtube_{self.split}.json")
        bbox_path = os.path.join(self.root_dir, f"youtube_{self.split}_bound_box.json")
        images_json_path = os.path.join(
            self.root_dir, f"youtube_{self.split}_images.json"
        )
        if os.path.exists(bbox_path) and os.path.exists(images_json_path):
            return read_json(bbox_path), read_json(images_json_path)
        else:
            data_json = read_json(data_json_path)
            images_dict = data_json["images"]
            save_json(images_dict, images_json_path)
            annotations_dict = data_json["annotations"]
            bbox = self.get_bbox_from_annotations(annotations_dict)
            save_json(bbox, bbox_path)
            return bbox, images_dict

    def get_bbox_from_annotations(self, annotations: dict) -> dict:
        """Converts vertices corresponding to mano mesh to 21 coordinates signifying
        the bound box.

        Args:
            annotations (dict): dictionary containing annotations.

        Returns:
            dict: same dictionary as annotations except 'vertices' is removed and
                'joints' key is added.
        """
        optimized_vertices = []
        for elem in tqdm(annotations):
            joints_21 = sudo_joint_bound(elem["vertices"])
            optimized_vertices.append(
                {
                    **{key: val for key, val in elem.items() if key != "vertices"},
                    **{"joints": joints_21.tolist()},
                }
            )
        return optimized_vertices

    def __len__(self):
        return len(self.bbox)

    def __getitem__(self, idx: int) -> dict:
        """Returns a sample corresponding to the index.

        Args:
            idx (int): index

        Returns:
            dict: item with following elements.
                "image" in opencv bgr format.
                "K": camera params
                "joints3D": 3D coordinates of joints in AIT format.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(
            self.root_dir, self.img_dict[self.bbox[idx]["image_id"]]["name"]
        )
        img = cv2.imread(img_name.replace(".png", ".jpg"))
        joints3D = torch.tensor(self.bbox[idx]["joints"]).float()
        joints3D[..., -1] = 1.0
        camera_param = torch.eye(3).float()
        joints_valid = torch.zeros_like(joints3D[..., -1])
        sample = {
            "image": img,
            "K": camera_param,
            "joints3D": joints3D,
            "joints_valid": torch.tensor(joints_valid),
        }
        return sample
