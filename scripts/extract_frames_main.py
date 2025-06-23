from utils.video_utils import save_first_frame, get_video_paths
import os
from utils.config_paths import FIRST_FRAMES_DIR, VIDEO_DIR

OUTPUT_PATH = FIRST_FRAMES_DIR
VIDEOS_PATH = VIDEO_DIR
os.makedirs(OUTPUT_PATH,exist_ok=True)

for video_path in  get_video_paths(VIDEOS_PATH):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_PATH, base_name + ".png")
    save_first_frame(video_path, output_path=output_path)