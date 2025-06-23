import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

INPUT_FOLDER_PATH = "/home/uros/Documents/project31/data/classifiers_frames/Talis_frames15_v2"
OUTPUT_FOLDER_PATH = "/home/uros/Documents/project31/data/classifiers_frames/depth_frames"
REPO_PATH = "/home/uros/Documents/other/dinov2/" # Specify a local path to the repository (or use installed package instead)
sys.path.append(REPO_PATH)

from dinov2.eval.depth.models import build_depther


## Utilities
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther


import os


def get_all_images(folder_path, extensions=['.jpg', '.jpeg', '.png', '.gif']):
    all_files = os.listdir(folder_path)
    images = [os.path.join(folder_path, f) for f in all_files if os.path.splitext(f)[1].lower() in extensions]
    return images


# load pretrained backbone
BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()

# load pretrained depth head

import urllib

import mmcv
from mmcv.runner import load_checkpoint


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_DATASET = "nyu" # in ("nyu", "kitti")
HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = create_depther(
    cfg,UsbHardDrive
# estimate depth on image
import matplotlib
from torchvision import transforms


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)



# get images in classifiers folder
image_files = get_all_images(INPUT_FOLDER_PATH)

# loop over the images
for image_path in image_files:
    image_id = os.path.basename(image_path)
    output_path_real = os.path.join(OUTPUT_FOLDER_PATH, "real_depth", image_id.split('.')[0] + ".npy")
    output_path_viz = os.path.join(OUTPUT_FOLDER_PATH, "vizualization", image_id.split('.')[0] + ".png")

    if image_id in os.listdir(os.path.join(OUTPUT_FOLDER_PATH, "vizualization")):
        print('pass')
        continue

    # load image
    image = Image.open(image_path).convert("RGB")


    # get depth image
    transform = make_depth_transform()

    scale_factor = 1
    rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)

    depth_image = render_depth(result.squeeze().cpu())

    # save image


    np.save(output_path_real, result.squeeze().cpu().numpy())

    # Using a colormap to convert depth to RGB

    depth_image.save(output_path_viz)