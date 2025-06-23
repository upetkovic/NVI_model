import os
import numpy as np
from PIL import Image

def save_colored_masks_from_npy(npy_path, output_mask_dir, palette):
    """
    Saves segmentation masks as colorized PNGs using a palette.

    Args:
        npy_path (str): Path to the .npy file with predicted masks.
        output_mask_dir (str): Directory to save the color PNGs.
        palette (list[int]): List of 768 RGB values (256 * 3) for palette coloring.
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    pred_masks = np.load(npy_path, allow_pickle=True)

    for frame_idx, mask in enumerate(pred_masks):
        mask_uint8 = mask.astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode="P")
        mask_pil.putpalette(palette)
        mask_rgb = mask_pil.convert("RGB")

        output_path = os.path.join(output_mask_dir, f"{frame_idx:05d}.png")
        mask_rgb.save(output_path)

    print(f"Saved {len(pred_masks)} color masks to {output_mask_dir}")


# Example palette: first few distinct colors, rest padded with black
_palette = [
    0, 0, 0,        # background
    255, 0, 0,      # red
    0, 255, 0,      # green
    0, 0, 255,      # blue
    255, 255, 0,    # yellow
    255, 0, 255,    # magenta
    0, 255, 255,    # cyan
    128, 128, 0,    # olive
    128, 0, 128,    # purple
    0, 128, 128     # teal
] + [0] * (768 - 30)  # pad to 768 values (256 classes)

# You can customize this list to match your visualization style
video_name = "00281"
npy_path = f"/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/semantic_segmentation/humans/real_segm7/{video_name}.npy"
output_mask_dir = f"./outputs/{video_name}_colored_masks"

save_colored_masks_from_npy(npy_path, output_mask_dir, _palette)
