from src.utils.video_utils import save_first_frame, get_video_paths
import os

VIDEOS_PATH = "data/inputs/videos"
OUTPUT_PATH = "./data/inputs/first_frames"

os.makedirs(OUTPUT_PATH,exist_ok=True)

for video_path in  get_video_paths(VIDEOS_PATH):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_PATH, base_name + ".jpg")
    save_first_frame(video_path, output_path=output_path)