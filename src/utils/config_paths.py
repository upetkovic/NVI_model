# config_paths.py

import yaml
from pathlib import Path

# Load config.yaml once
with open("data/config/config_paths.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Define variables to be imported
VIDEO_DIR = Path(cfg["video_dir"])
OUTPUTS_MAIN_FOLDER = Path(cfg["output_dir"])
json_first_frame_labels_path = Path(cfg["json_first_frame_labels_path"])
FIRST_FRAMES_DIR = Path(cfg["first_frames_dir"])
