from utils.video_utils import resample_video, get_video_paths
import os

VIDEOS_INPUT = "./data/inputs/videos"
OUTPUT_PATH = "./data/outputs/resampled_videos"

os.makedirs(OUTPUT_PATH, exist_ok=True)
for video_path in get_video_paths(VIDEOS_INPUT):

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_PATH, base_name + ".avi")    
    resample_video(video_path, output_path, frame_step=7)