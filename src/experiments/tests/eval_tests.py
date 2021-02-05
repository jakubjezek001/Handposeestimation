from src.data_loader.utils import get_data
from src.experiments.evaluation_utils import calc_procrustes_transform
from math import cos, sin
import unittest

import torch


def get_rot_mat(angle):
    return torch.tensor(
        [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
    )


class TestExperimentMethods(unittest.TestCase):
    def test_calc_procrustes_transform(self):
        print("Perfroming test on procrsutes transform")
        joints1 = torch.rand((128, 21, 3))
        translate, scale, angle = torch.tensor([5.0, 6.0, 0.0]), 5.0, 90.0
        rot_mat = get_rot_mat(angle).view(1, 3, 3).repeat(128, 1, 1)
        joints2 = joints1.clone()
        joints2 = torch.bmm(joints2, rot_mat.transpose(2, 1)) * scale
        joints2[..., 0] += translate[0]
        joints2[..., 1] += translate[1]
        joints2[..., 2] += translate[2]

        (
            joints_transformed,
            rot_cal,
            scale_cal,
            translation_cal,
        ) = calc_procrustes_transform(joints1, joints2)
        self.assertTrue(((joints_transformed - joints1).abs().max() < 1e-5).tolist())
        # calculated rotation matrix is in batch transposed format.
        self.assertTrue(
            (
                (rot_cal - get_rot_mat(-angle).T.view(1, 3, 3)).abs().max() < 1e-6
            ).tolist()
        )
        self.assertTrue(((scale - 1 / scale_cal).abs().max() < 1e-3).tolist())
        self.assertTrue(
            (
                (
                    translation_cal
                    + (get_rot_mat(-angle) @ translate.view(3, 1) / scale).view(1, 1, 3)
                )
                .abs()
                .max()
                < 1e-5
            ).tolist()
        )


if __name__ == "__main__":
    unittest.main()
