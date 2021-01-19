from easydict import EasyDict as edict
from src.constants import HRNET_CONFIG
from src.models.external.HRnet.pose_hrnet import get_pose_net
from src.models.unsupervised.simclr_model import SimCLR
from src.utils import read_yaml
from torch import nn


class SimCLRHeatmap(SimCLR):
    """
    SimcLR implementation inspired from paper https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    """

    def __init__(self, config: edict):
        super().__init__(config)
        self.epsilon = 1e-6
        hrnet_config = read_yaml(HRNET_CONFIG)
        self.encoder = get_pose_net(hrnet_config.MODEL36, True)

    def get_projection_head(self) -> nn.Sequential:
        projection_head = nn.Sequential(
            nn.Conv2d(42, 32, kernel_size=(8, 8), stride=8),
            nn.Flatten(),
            nn.Linear(
                self.config.projection_head_input_dim,
                self.config.projection_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.projection_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.projection_head_hidden_dim,
                self.config.output_dim,
                bias=False,
            ),
        )
        return projection_head
