import os
from easydict import EasyDict as edict
import numpy as np
from src.types import JOINTS_3D
from src.utils import read_json
from src.constants import MASTER_THESIS_DIR


class Joints:
    def __init__(self):
        self.mapping = edict(
            read_json(
                os.path.join(
                    MASTER_THESIS_DIR, "src", "data_loader", "joint_mapping.json"
                )
            )
        )
        self.freihand_ait_index_map = self.get_set1_to_set2_index_map(
            set1="freihand", set2="ait"
        )
        self.ait_freihand_index_map = self.get_set1_to_set2_index_map(
            set1="ait", set2="freihand"
        )

    def get_set1_to_set2_index_map(
        self, set1: str = "freihand", set2: str = "ait"
    ) -> np.array:
        index_map = []
        for i in self.mapping.ait.keys():
            index_map.append([self.mapping[set1][i], self.mapping[set2][i]])
        return np.array(sorted(index_map, key=lambda x: x[0]))

    def get_freihand_ait_index_mapping(self) -> np.array:
        index_map = []
        for i in self.mapping.ait.keys():
            index_map.append([self.mapping.freihand[i], self.mapping.ait[i]])
        return np.array(sorted(index_map, key=lambda x: x[0]))

    def freihand_to_ait(self, joints_3D: JOINTS_3D) -> JOINTS_3D:
        return joints_3D[self.freihand_ait_index_map[:, 1]]

    def ait_to_freihand(self, joints_3D: JOINTS_3D) -> JOINTS_3D:
        return joints_3D[self.ait_freihand_index_map[:, 1]]
