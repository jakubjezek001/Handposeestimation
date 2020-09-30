from src.data_loader.data_set import Data_Set
from src.utils import read_json
from easydict import EasyDict as edict
from src.constants import TRAINING_CONFIG_PATH
from src.data_loader.utils import convert_2_5D_to_3D, error_in_conversion
from tqdm import tqdm


def main():
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    data = Data_Set(config=train_param, transforms=None, train_set=True)
    for id in tqdm(range(len(data))):
        joints25D = data[id]["joints"]
        scale = data[id]["scale"]
        K = data[id]["K"]
        true_joints_3D = data[id]["joints_3D"]
        cal_joints_3D = convert_2_5D_to_3D(joints25D, scale, K)
        error = error_in_conversion(true_joints_3D, cal_joints_3D)
        if error > 1e-3:
            print(f"High error found {error} of the true ")
            break


if __name__ == "__main__":
    main()
