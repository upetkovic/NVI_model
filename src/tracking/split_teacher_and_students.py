import json
import glob
import os
from utils.video_utils import get_images_paths
import cv2
import numpy as np

def get_json_path(search_dir):
    json_files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.endswith(".json")]

    # 3. Check how many were found
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON file found in {search_dir}")
    elif len(json_files) > 1:
        raise RuntimeError(f"Multiple JSON files found in {search_dir}: {json_files}")

    # 4. If exactly one found
    json_path = json_files[0]

    return json_path    

def split_teacher_and_students(json_path, masks_paths, teacher_folder_path, video_id):
    # 1. Load the JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2. Build the new simplified dictionary
    filename_to_point = {}

    for entry in data.values():
        filename = entry["filename"]
        region = entry["regions"][0]  # take the first region
        cx = region["shape_attributes"]["cx"]
        cy = region["shape_attributes"]["cy"]
        filename_to_point[filename] = (cx, cy)
    #teacher_folder_path = os.path.join(tracking_folder, video_id + "_teacher")
    os.makedirs(teacher_folder_path, exist_ok=True)
    cx, cy = filename_to_point[video_id+".png"]
    masks_paths = get_images_paths(masks_paths)
    masks_paths.sort()
    mask0 = cv2.imread(masks_paths[0])
    color = mask0[cy, cx]
    color = tuple(color)

    for mask_path in masks_paths:
        basename = os.path.basename(mask_path)
        mask = cv2.imread(mask_path)
        object_mask = np.all(mask == color, axis=-1).astype(np.uint8)
        binary_mask = object_mask * 255
        cv2.imwrite(os.path.join(teacher_folder_path, basename), binary_mask)


if __name__ == "__main__":

    """
    split_teacher_and_students(
        json_folder="data/inputs/first_frames",
        tracking_folder="./data/outputs/tracking",
        video_id="00281"
    )
    
    split_teacher_and_students(
        json_path="data/inputs/first_frames/via_project_6May2025_16h7m_json.json",
        masks_paths="outputs/00281_colored_masks", 
        teacher_folder_path="outputs/00281_colored_teacher",
        video_id="00281"
        )
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to the json file.")
    parser.add_argument("--tracking_folder", required=True, help="Path to the tracking folder with masks.")
    parser.add_argument("--teacher_folder_path", required=True, help="Output path to the teacher folder.")
    parser.add_argument("--video_id", required=True, help="Basename of video")
    args = parser.parse_args()

    split_teacher_and_students(
        json_path=args.json_path,
        masks_paths=args.tracking_folder,
        teacher_folder_path=args.teacher_folder_path,
        video_id=args.video_id
    )
