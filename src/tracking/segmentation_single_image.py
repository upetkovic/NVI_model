import os
import sys
import torch
import gc
import cv2
import numpy as np

def segment_single_image(
    image_path, 
    grounding_caption="humans", 
    box_threshold=0.35, 
    text_threshold=0.5, 
    box_size_threshold=0.5, 
    reset_image=True
):
    """
    Segments a single image using SegTracker and returns the predicted mask and annotated frame.

    Args:
        image_path (str): Path to the input image.
        grounding_caption (str): Text prompt to guide segmentation.
        box_threshold (float): Threshold for box detection.
        text_threshold (float): Threshold for text detection.
        box_size_threshold (float): Box size threshold to filter boxes.
        reset_image (bool): Whether to reset image embedding.

    Returns:
        pred_mask (np.ndarray): Predicted segmentation mask.
        annotated_frame (np.ndarray): Frame with visualized mask.
        boxes (list): List of detected bounding boxes.
    """

    # Add project root and external folders
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything/aot"))

    from SegTracker import SegTracker
    from model_args import aot_args, sam_args, segtracker_args
    from aot_tracker import _palette
    from utils.helpers import get_iou
    #from your_script import draw_mask  # make sure draw_mask is accessible
    from tracking.segementation_and_track import draw_mask
    # Read image
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Modify sam_args if needed
    sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }

    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()

    with torch.cuda.amp.autocast():
        pred_mask, annotated_frame, boxes = segtracker.detect_and_seg_mod(
            frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image
        )

    masked_frame = draw_mask(frame,pred_mask)
    masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

    return pred_mask, masked_frame, boxes


if __name__ == "__main__":
    image_path = "data/outputs/first_frames/00152r.jpg"
    for i in range(1):
        print(i)
        pred_mask, annotated_frame, boxes = segment_single_image(image_path)
        print("-"*25)
    # visualize/save result
    import matplotlib.pyplot as plt
    plt.imshow(annotated_frame)
    plt.axis('off')
    plt.show()
