from src.experiments.evaluation_utils import calc_procrustes_transform

import unittest

import torch


class TestExperimentMethods(unittest.TestCase):
    def test_calc_procrustes_transform(self):
        print("Perfroming test on procrsutes transform")
        joints1 = torch.rand((128, 21, 3))
        translate, scale = torch.tensor([5.0, 6.0, 0.0]), 5.0
        joints2 = joints1.clone() * scale
        joints2[..., 0] += translate[0]
        joints2[..., 1] += translate[1]
        joints2[..., 2] += translate[2]
        (
            joints_transformed,
            rot_cal,
            scale_cal,
            translation_cal,
        ) = calc_procrustes_transform(joints1, joints2)
        true_rot = torch.eye(3).view(1, 3, 3)
        # print((scale - 1/scale_cal).abs().max())
        self.assertTrue(((joints_transformed - joints1).abs().max() < 1e-6).tolist())
        self.assertTrue(((rot_cal - true_rot).abs().max() < 1e-6).tolist())
        self.assertTrue(((scale - 1 / scale_cal).abs().max() < 1e-3).tolist())

        self.assertTrue(
            (
                (translation_cal + translate.view(1, 1, 3) / scale).abs().max() < 1e-5
            ).tolist()
        )


if __name__ == "__main__":
    unittest.main()
