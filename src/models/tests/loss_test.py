import unittest

import torch
from src.models.utils import cal_l1_loss


class TestStringMethods(unittest.TestCase):
    def test_cal_l1_loss(self):
        pred_joints = torch.ones((12, 21, 3), dtype=torch.float16)
        true_joints = torch.ones((12, 21, 3), dtype=torch.float16) * 2
        scale = torch.ones((12), dtype=torch.float16) * 10.0
        joints_valid = torch.ones((12, 21, 1), dtype=torch.float16)
        joints_valid[10, 5:20] = 0.0
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(
            pred_joints, true_joints, scale, joints_valid
        )
        self.assertTrue(((loss_z - 1.0) < 1e-6).tolist())
        self.assertTrue(((loss_2d - 1.0) < 1e-6).tolist())
        self.assertTrue(((loss_z_unscaled - 10.0) < 1e-6).tolist())


if __name__ == "__main__":
    unittest.main()
