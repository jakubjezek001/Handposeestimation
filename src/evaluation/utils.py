from typing import Dict, Union

import matplotlib.pyplot as plt
import torch
from easydict import EasyDict as edict
from src.constants import SUPERVISED_CONFIG_PATH
from src.data_loader.joints import Joints
from src.data_loader.sample_augmenter import SampleAugmenter
from src.data_loader.utils import convert_2_5D_to_3D
from src.experiments.utils import restore_model
from src.models.supervised import BaselineModel, DenoisedBaselineModel
from src.utils import read_json
from torchvision import transforms

BBOX_SCALE = 0.33
CROP_SIZE = 128
JOINTS = Joints()


def load_model(
    key: str, resnet_size: str
) -> Union[BaselineModel, DenoisedBaselineModel]:
    """Loads saved model given a key, so far only resnet style models are supported.

    Args:
        key (str): Experiment key

    Returns:
        Union[BaselineModel, DenoisedBaselineModel]: saved model, in eval mode and on GPU.
    """
    dev = torch.device("cuda")
    model_config = edict(read_json(SUPERVISED_CONFIG_PATH))
    model_config.resnet_size = resnet_size
    print(f"Loading latest checkpoint of {key}")
    try:
        print("Trying DEnoised model!")
        model = DenoisedBaselineModel(model_config)
        model = restore_model(model, key, "")
    except Exception as e:
        print(e)
        print("Trying Baseline model!")
        try:
            model = BaselineModel(model_config)
            model = restore_model(model, key, "")
            print("Model loaded successfully!")
        except Exception as k:
            print(k)
            print(f"Experiment {key} not found !")
            return None
    model.eval()
    model.to(dev)
    return model


def normalize_joints(joints3d: torch.Tensor) -> torch.Tensor:
    """Scales the joints such that bone between middle_mcp and middle_pip is unit length because this lenght is
    provided in freihand data and data_set class scales it

    Args:
        joints3d ( torch.Tensor): [description]

    Returns:
        ( torch.Tensor): Normalized joints with respect to phalangal proximal bone of the middle finger.
    """
    return joints3d / torch.linalg.norm(
        joints3d[JOINTS.mapping.ait.middle_mcp]
        - joints3d[JOINTS.mapping.ait.middle_pip]
    )


def create_sudo_bound_box(scale: float) -> torch.Tensor:
    """Creates a square bound box around image center. scale controls the ratio by which the square is scaled.
    Scale 1.0 will return boundbox the size of freihand image.

    Args:
        scale (float): Scale factor for boundbox size

    Returns:
        torch.Tensor: A bound box of size [21,3]. The z axis is 1 and first two dimensions are corner points
            of boundbox in pixels.
    """
    max_bound = torch.tensor([224.0, 224.0])
    min_bound = torch.tensor([0.0, 0.0])
    c = (max_bound + min_bound) / 2.0
    s = ((max_bound - min_bound) / 2.0) * scale
    bound_box = torch.tensor(
        [[0, 0, 0]]
        + [[s[0], s[1], 1]] * 5
        + [[-s[0], s[1], 1]] * 5
        + [[s[0], -s[1], 1]] * 5
        + [[-s[0], -s[1], 1]] * 5
    ) + torch.tensor([c[0], c[1], 0])
    return bound_box.float()


def process_data(
    sample: dict,
    bbox: torch.Tensor,
    augmenter: SampleAugmenter,
    transform: transforms.Compose,
    step=1,
) -> Dict[str, torch.Tensor]:
    """Processes the images according to augmenter and transfromer. The boundbox acts as proxy for 2D joints
    for efficienct cropping.

    Args:
        sample (dict): dictionary containing original image("image") and camera intrinsics (K)
        bbox (torch.Tensor): 21x3 dimensional boundbox
        augmenter (SampleAugmenter): Augmenter to augment sample, used for cropping and resizing
        transform (transforms.Compose): Transform to be performed on image, like conversion to tensor and
            normalization
        step (int, optional): variable for debugging the image processing step. Defaults to 1.

    Returns:
        Dict[str, torch.Tensor]:  Dictionary containg augemented "image" , adapted camera intrinsics("K") and
            "transformation_matrix"
    """
    image, _, transformation_matrix = augmenter.transform_sample(sample["image"], bbox)
    # plt.imshow(image)
    # plt.show()
    # plt.savefig(f"image_{step}.png")
    image = transform(image)
    transformation_matrix = torch.tensor(transformation_matrix).float()

    return {
        "image": image.view([1] + list(image.size())),
        "K": transformation_matrix @ sample["K"],
        "transformation_matrix": transformation_matrix,
    }


def model_refined_inference(
    model: Union[BaselineModel, DenoisedBaselineModel],
    sample: dict,
    augmenter: SampleAugmenter,
    transform: transforms.Compose,
) -> torch.Tensor:
    """Calculates refined bound box from an initial estimate around image center and uses that bound box to
    predict the joints3D.

    Args:
        model (Union[BaselineModel, DenoisedBaselineModel]): Trained model
        sample (dict): image and camera intrinsics dictionary
        augmenter (SampleAugmenter): augmenter for processing image(cropping and resizing)
        transform (transforms.Compose): Transforms on image, normalization and tensor conversion.

    Returns:
        torch.Tensor: predicted I3D joints
    """
    img_orig, K = sample["image"], sample["K"]
    sudo_bbox = create_sudo_bound_box(BBOX_SCALE)
    sample = process_data(
        {"image": img_orig.copy(), "K": K.clone()}, sudo_bbox, augmenter, transform, 1
    )
    predictions25d = model(sample["image"].to(model.device)).view(21, 3)
    predictions25d[..., -1] = 1.0
    bbox = (
        predictions25d
        @ torch.inverse(sample["transformation_matrix"].to(model.device)).T
    )
    # Cropping image with refined crop box.
    sample = process_data(
        {"image": img_orig.copy(), "K": K.clone()}, bbox, augmenter, transform, step=2
    )
    predictions25d = model(sample["image"].to(model.device)).view(21, 3)
    if hasattr(model, "denoiser"):
        z_root_calc_denoised = model.get_denoised_z_root_calc(
            predictions25d.view(1, 21, 3), sample["K"].view(1, 3, 3).to(model.device)
        )
        predictions3d = convert_2_5D_to_3D(
            predictions25d.cpu(),
            1.0,
            sample["K"].cpu(),
            Z_root_calc=z_root_calc_denoised.cpu().view(-1),
        )
    else:
        predictions3d = convert_2_5D_to_3D(predictions25d.cpu(), 1.0, sample["K"].cpu())

    return predictions3d
