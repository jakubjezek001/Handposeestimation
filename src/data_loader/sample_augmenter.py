import random
from typing import Tuple

import cv2
import numpy as np
import torch
from src.types import JOINTS_25D


class SampleAugmenter:
    def __init__(self, crop: bool, resize: bool, rotate: bool, seed: int, **kwargs):
        """Initialization of the sample augmentor class.

        Args:
            crop (bool): To crop the image around keypoints/joints
            resize (bool): To resize the image according to 'resize_shape' kwargs. Default is (128,128).
            rotate (bool): To rotate the image randomly
            seed (int): Seed for random rotation.
        """
        self.crop = crop
        self.resize = resize
        self.rotate = rotate
        random.seed(seed)
        if "crop_margin" in kwargs:
            self.crop_margin = kwargs["crop_margin"]
        else:
            self.crop_margin = 1.5
        if "resize_shape" in kwargs:
            self.resize_shape = tuple(kwargs["resize_shape"])
        else:
            self.resize_shape = (128, 128)  # (width, height)

    def get_rotation_matrix(self, center: Tuple[int, int]) -> np.array:
        """Function to get the roation matrix according to randomly sampled angle.

        Args:
            center (Tuple[int, int]): center coordinates (x, y)

        Returns:
            np.array: A 2 x 3 rotation matrix.
        """
        angle = random.uniform(-90, 90) // 1
        return cv2.getRotationMatrix2D(center, angle, 1.0)

    def get_crop_size(self, joints: JOINTS_25D) -> Tuple[int, int, int]:
        """Function to obtain the top left corner of the crop square and the side.

        Args:
            joints (JOINTS_25D): 2.5D joints Only 2D image coordinates are used.

        Returns:
            Tuple[int, int, int]:  Top left coordinates of the crop box and the side of the crop box.
        """
        top, left = torch.min(joints[:, 1]), torch.min(joints[:, 0])
        bottom, right = torch.max(joints[:, 1]), torch.max(joints[:, 0])
        height, width = bottom - top, right - left
        side = int(max(height, width) * self.crop_margin)
        origin_x = int(left - width * (self.crop_margin - 1) / 2)
        origin_y = int(top - height * (self.crop_margin - 1) / 2)
        return origin_x, origin_y, side

    def crop_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        """Crops the sample around a crop box conataining all key points..

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): 2.5D joints. The depth is kept as is.

        Returns:
            Tuple[np.array JOINTS_25D]: cropped image and adjusted keypoints.
        """
        origin_x, origin_y, side = self.get_crop_size(joints)
        joints[:, 0] = joints[:, 0] - origin_x
        joints[:, 1] = joints[:, 1] - origin_y
        return image[origin_y : origin_y + side, origin_x : origin_x + side, :], joints

    def resize_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        """Resizes the sample to given size.

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): 2.5D joints. The depth is kept as is.

        Returns:
            Tuple[np.array JOINTS_25D]: Resized image and keypoints.
        """
        height, width = image.shape[:2]
        image = cv2.resize(image, self.resize_shape, interpolation=cv2.INTER_AREA)
        joints[:, 0] = joints[:, 0] * self.resize_shape[0] / width
        joints[:, 1] = joints[:, 1] * self.resize_shape[1] / height
        return image, joints

    def rotate_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        """Rotates the sample image and the 2D keypoints by a random angle. The  relative depth is not changed.

        Args:
            image (np.array): an Image as a numpy array, preferable uncropped
            joints (JOINTS_25D): Tensor of all 2.5 D coordinates.

        Returns:
            Tuple[np.array, JOINTS_25D]: Rotated image and keypoints.
        """
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rot_mat = self.get_rotation_matrix(center=center)
        image = cv2.warpAffine(image, rot_mat, (width, height))
        joints_ = joints.clone()
        joints_[:, -1] = 1.0
        joints_ = joints_ @ rot_mat.T
        joints[:, :-1] = joints_
        return image, joints

    def transform_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        """Transforms  the sample image and the 2D keypoints. The  relative depth is not changed.

        Args:
            image (np.array): an Image as a numpy array, preferable uncropped
            joints (JOINTS_25D): Tensor of all 2.5 D coordinates.

        Returns:
            Tuple[np.array, JOINTS_25D]: Transformed image and keypoints.
        """
        image_, joints_ = image.copy(), joints.clone()
        if self.rotate:
            image_, joints_ = self.rotate_sample(image_, joints_)
        if self.crop:
            image_, joints_ = self.crop_sample(image_, joints_)
        if self.resize:
            image_, joints_ = self.resize_sample(image_, joints_)
        return image_, joints_
