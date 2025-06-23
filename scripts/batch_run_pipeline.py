import os
import subprocess
import glob
from tqdm import tqdm
from utils.config_paths import VIDEO_DIR

# Path to your single-video pipeline script
PIPELINE_SCRIPT = "scripts/run_pipeline_for_video.py"

# Folder containing all videos to process
#VIDEO_FOLDER = "/home/uros/Documents/project31/data/Talis_teachers_shorten_clean"

ENV = "nvi_env2"

# Get all .avi video files
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))

print(f"Found {len(video_files)} videos.")

for video_path in tqdm(video_files, desc="Processing videos"):
    print("\n" + "="*20)
    print(f"Processing video: {video_path}")
    print("="*20)

    # Call the pipeline script using conda run, pass video path as argument
    command = f'conda run -n {ENV} python {PIPELINE_SCRIPT} --video_path "{video_path}"'
    subprocess.run(command, shell=True)
    