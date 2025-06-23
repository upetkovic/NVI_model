import numpy as np
import os
import json

def compute_mask_box_iou(mask, box):
    """
    Compute IoU between binary mask and bounding box.
    - mask: 2D binary numpy array (0 or 1)
    - box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = mask.shape

    # Clamp box to image bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    box_mask = np.zeros_like(mask, dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 1

    intersection = np.logical_and(mask, box_mask).sum()
    union = np.logical_or(mask, box_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def get_iou(bbox0, bbox1):
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox0[0][0], bbox1[0][0])
    y_top = max(bbox0[0][1], bbox1[0][1])
    x_right = min(bbox0[1][0], bbox1[1][0])
    y_bottom = min(bbox0[1][1], bbox1[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Compute the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    bbox0_area = (bbox0[1][0] - bbox0[0][0]) * (bbox0[1][1] - bbox0[0][1])
    bbox1_area = (bbox1[1][0] - bbox1[0][0]) * (bbox1[1][1] - bbox1[0][1])

    # Compute the IoU
    iou = intersection_area / float(bbox0_area + bbox1_area - intersection_area)

    return iou


def create_command(script, environment, **arguments):
    """
    Creates a command string to execute a Python script with the given arguments
    using `conda run` to properly isolate environments in subprocesses.
    """
    args = " ".join(f"--{k} {v}" for k, v in arguments.items())
    return f"conda run -n {environment} python {script} {args}"

def get_first_json_file(directory):
    """
    Returns the content of the first JSON file found in the given directory.
    If no JSON file is found, returns None.
    """
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            return os.path.join(directory, file_name)
    
    return None