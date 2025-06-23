import os
import sys
#from utils.video_utils import get_video_paths

# Add project root and required folders to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)  # Project root
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything"))  # Segment & Track Anything folder
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything/aot"))  # AOT folder

# Now imports can be found
#from utils.helpers import get_iou  # Uncomment this if you need it
import cv2
import numpy as np
import torch
import gc
from tqdm import tqdm
from PIL import Image
from SegTracker import SegTracker
from model_args import aot_args, sam_args, segtracker_args
from aot_tracker import _palette



def save_prediction(pred_mask, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))


def draw_mask(img, mask, alpha=0.7):
    colored_mask = Image.fromarray(mask.astype(np.uint8))
    colored_mask = colored_mask.convert(mode='P')
    colored_mask.putpalette(_palette)
    colored_mask = colored_mask.convert(mode='RGB')
    foreground = np.array(colored_mask)
    blended = cv2.addWeighted(img, 1 - alpha, foreground, alpha, 0)
    return blended


def track_and_segment_video(
    input_video_path,
    output_folder,
    grounding_caption="humans",
    sampling_freq=7,
    box_threshold=0.35,
    text_threshold=0.5,
    box_size_threshold=0.5,
    reset_image=True,
):
    """
    Tracks and segments objects in a video using SegTracker.

    Args:
        input_video_path (str): Path to input video.
        output_folder (str): Directory to save outputs.
    """
    os.makedirs(output_folder, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    output_mask_dir = os.path.join(output_folder, f"{video_name}_masks")
    #output_mask_dir = os.path.join(output_folder, f"segm_masks")

    output_video_path = os.path.join(output_folder, f"{video_name}_seg.mp4")

    os.makedirs(output_mask_dir, exist_ok=True)
    output_path_mask_npy = os.path.join(output_mask_dir, video_name + ".npy")

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps // sampling_freq, (width, height))

    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()

    frame_idx = 0
    pred_masks = []

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                pred_mask, annotated_frame, _ = segtracker.detect_and_seg_mod(
                    frame_rgb, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image
                )
                segtracker.add_reference(frame_rgb, pred_mask)
            elif frame_idx % segtracker_args['sam_gap'] == 0:
                seg_mask, _ = segtracker.detect_and_seg(
                    frame_rgb, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image
                )
                track_mask = segtracker.track(frame_rgb)
                new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                pred_mask = track_mask + new_obj_mask
                segtracker.add_reference(frame_rgb, pred_mask)
            else:
                pred_mask = segtracker.track(frame_rgb, update_memory=True)

            save_prediction(pred_mask, output_mask_dir, f"{frame_idx:05d}.png")

            if frame_idx % sampling_freq == 0:
                blended_frame = draw_mask(frame_rgb, pred_mask)
                blended_frame_bgr = cv2.cvtColor(blended_frame, cv2.COLOR_RGB2BGR)
                out_video.write(blended_frame_bgr)

            pred_masks.append(pred_mask)
            frame_idx += 1

            torch.cuda.empty_cache()
            gc.collect()

    cap.release()
    out_video.release()
    np.save(output_path_mask_npy, pred_masks)
    print(f"Saved segmented video to {output_video_path}")
    print(f"Numpy mask saved to {output_path_mask_npy}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True, help="Path to input video.")
    parser.add_argument("--output_folder", required=True, help="Folder to save outputs.")
    args = parser.parse_args()

    track_and_segment_video(
        input_video_path=args.input_video,
        output_folder=args.output_folder,
    )