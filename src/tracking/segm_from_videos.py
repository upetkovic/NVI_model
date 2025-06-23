import cv2
import os
import sys

# Add root of project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

# Add external folders explicitly
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything/aot"))


from external.segment_and_track_anything.SegTracker import SegTracker
from external.segment_and_track_anything.SegTracker import SegTracker
from external.segment_and_track_anything.model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from external.segment_and_track_anything.aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.7, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

def get_all_images(folder_path, extensions=['.jpg', '.jpeg', '.png', '.gif']):
    all_files = os.listdir(folder_path)
    images = [os.path.join(folder_path, f) for f in all_files if os.path.splitext(f)[1].lower() in extensions]
    return images

def reduce_frame_resolution(orig_frame, scale_percent=50):
    """
    Reduce the resolution of a given frame.
    
    Parameters:
    - orig_frame: The original frame to resize.
    - scale_percent: Percentage of the original size. Default is 50%.
    
    Returns:
    - Resized frame.
    """
    
    # Calculate the dimensions for the resized frame
    width = int(orig_frame.shape[1] * scale_percent / 100)
    height = int(orig_frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # Resize the frame
    resized_frame = cv2.resize(orig_frame, dim, interpolation=cv2.INTER_AREA)
    
    return resized_frame

frame_reduce_factor = 7
INPUT_FOLDER_PATH = "data/inputs/videos"
OUTPUT_FOLDER_PATH = f"data/outputs/videos"
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)
grounding_caption = "humans"
box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True

sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }

# get images in classifiers folder
video_files = get_all_images(INPUT_FOLDER_PATH, extensions=[".avi"])
segtracker = SegTracker(segtracker_args,sam_args,aot_args)
# loop over the images
segtracker = SegTracker(segtracker_args, sam_args, aot_args)
for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_id = os.path.basename(video_path)
    # Create folder to save visualization frames
    video_name_no_ext = os.path.splitext(video_id)[0]
    viz_frame_folder = os.path.join(OUTPUT_FOLDER_PATH, "tracking", video_name_no_ext)
    os.makedirs(viz_frame_folder, exist_ok=True)

    output_path_real = os.path.join(OUTPUT_FOLDER_PATH, video_id.split('.')[0] + ".npy")
    output_path_viz = os.path.join(OUTPUT_FOLDER_PATH, "vizualization", video_id.split('.')[0] + ".png")

    if os.path.isfile(output_path_real):
        print('pass')
        continue


    pred_mask_list = []
    init_res_list = []
    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = segtracker_args['sam_gap']
    frame_idx = 0

    
    segtracker.restart_tracker()
    
    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_reduce_factor == 0:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #frame = reduce_frame_resolution(frame, scale_percent=50)

                pred_mask, annotated_frame = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image=True)
                obj_ids = np.unique(pred_mask)
                obj_ids = obj_ids[obj_ids!=0]           

                init_res = draw_mask(annotated_frame, pred_mask,id_countour=False)
                #init_res_list.append(init_res)
                pred_mask_list.append(pred_mask)

                # Save visualized frame
                output_frame_path = os.path.join(viz_frame_folder, f"{frame_idx:05d}.jpg")
                init_res_bgr = cv2.cvtColor(init_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_frame_path, init_res_bgr)

                torch.cuda.empty_cache()
                gc.collect()

                print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
            frame_idx += 1

        
        cap.release()
    

    np.save(output_path_real, pred_mask_list)
