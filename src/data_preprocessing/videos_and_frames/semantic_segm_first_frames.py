import os
import sys

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

# Add external root (only once!)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external"))

# Now imports
from tracking.segmentation_single_image import segment_single_image
from utils.video_utils import get_images_paths
import cv2

FIRST_FRAMES_INPUT = "data/outputs/first_frames"
OUTPUT_PATH = "./data/inputs/first_frames_segmentation"
os.makedirs(OUTPUT_PATH, exist_ok=True)

for image_path in get_images_paths(FIRST_FRAMES_INPUT):
    _, masked_frame, _ = segment_single_image(image_path)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(OUTPUT_PATH, base_name + ".jpg")    
    cv2.imwrite(output_path, masked_frame)
