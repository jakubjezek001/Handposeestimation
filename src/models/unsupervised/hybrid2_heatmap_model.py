from typing import Dict

from easydict import EasyDict as edict
from src.models.unsupervised.hybrid2_model import Hybrid2Model
from src.models.unsupervised.simclr_heatmap_model import SimCLRHeatmap
from torch import Tensor


class Hybrid2HeatmapModel(SimCLRHeatmap, Hybrid2Model):
    """
    Hybrid2 version for simCLRr implementation inspired from paper
    https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    The equivariance is preserved by transforming projection space.
    """

    def __init__(self, config: edict):
        SimCLRHeatmap.__init__(self, config)

    def contrastive_step(self, batch: Dict[str, Tensor]) -> Tensor:
        return Hybrid2Model.contrastive_step(self, batch)
