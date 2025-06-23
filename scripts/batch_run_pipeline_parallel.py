import os
import subprocess
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from threading import Semaphore
from utils.config_paths import json_first_frame_labels_path, OUTPUTS_MAIN_FOLDER, VIDEO_DIR

# Configuration
PIPELINE_SCRIPT = "scripts/run_pipeline_for_video.py"
PIPELINE_SCRIPT = "scripts/run_pipeline_for_video_v2.py"

VIDEO_FOLDER = VIDEO_DIR
ENV = "nvi_env"
GPU_IDS = [0]
PROCESSES_PER_GPU = 8

# Get all video files
video_files = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.avi")))
print(f"Found {len(video_files)} videos.")

# Create a semaphore for each GPU to limit concurrent processes
gpu_semaphores = {gpu_id: Semaphore(PROCESSES_PER_GPU) for gpu_id in GPU_IDS}

# Round-robin GPU assignment
gpu_assignment = zip(video_files, cycle(GPU_IDS))

def process_video(video_path, gpu_id):
    semaphore = gpu_semaphores[gpu_id]
    with semaphore:  # limit concurrent GPU usage
        print("\n" + "="*20)
        print(f"Processing video: {video_path} on GPU {gpu_id}")
        print("="*20)

        command = f'CUDA_VISIBLE_DEVICES={gpu_id} conda run -n {ENV} python {PIPELINE_SCRIPT} --video_path "{video_path}"'
        subprocess.run(command, shell=True)

# Use ThreadPoolExecutor (or ProcessPoolExecutor if CPU-bound prep is heavy)
with ThreadPoolExecutor(max_workers=len(GPU_IDS) * PROCESSES_PER_GPU) as executor:
    futures = [
        executor.submit(process_video, video_path, gpu_id)
        for video_path, gpu_id in gpu_assignment
    ]

    for _ in tqdm(as_completed(futures), total=len(video_files), desc="Processing videos"):
        pass
