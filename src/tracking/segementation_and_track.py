import os
import sys

# Add root of project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

# Add external folders explicitly
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external/segment_and_track_anything/aot"))
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image, ImageDraw
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
from utils.helpers import get_iou
import pandas as pd
import csv
#from tool.my_tools import get_bbox_from_masked_frame

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))

def save_prediction2(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    #save_mask = save_mask.convert(mode='P')
    #save_mask.putpalette(_palette)
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


def get_bbox_from_masked_frame(masked_frame):
    # Get the indices of non-zero pixels
    ys, xs = np.where(masked_frame > 0)

    # Check if there are any non-zero pixels
    if len(xs) == 0 or len(ys) == 0:
        return None  # No bounding box can be obtained

    # Get bounding box coordinates
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    # Return as a 2D list
    return [[x_min, y_min], [x_max, y_max]]


def get_best_match_bbox(reference_bbox, bbox_list):
    """
    Compares a reference bbox with a list of bboxes and returns the bbox from the list 
    that has the highest IoU with the reference bbox.

    Args:
        reference_bbox: The bounding box to compare against.
        bbox_list: A list of bounding boxes to compare.

    Returns:
        The bounding box from bbox_list with the highest IoU compared to reference_bbox.
    """
    max_iou = 0
    best_bbox = None

    for bbox in bbox_list:
        iou = get_iou(reference_bbox, bbox)
        if iou > max_iou:
            max_iou = iou
            best_bbox = bbox

    return best_bbox, max_iou

def draw_and_save_bbox(frame, bbox, output_path):
    """
    Draws a rectangle (bbox) over an image (numpy array) and saves the resultant image.

    Args:
        frame: A numpy array representing an image.
        bbox: A numpy array with the shape (2, 2) representing the bounding box.
        output_path: Path to save the resultant image.

    Returns:
        None
    """
    # Convert the numpy array to a PIL Image
    img = Image.fromarray(frame)
    
    # Create a drawing context for the image
    draw = ImageDraw.Draw(img)
    
    # Unpack the bounding box coordinates
    (x_min, y_min), (x_max, y_max) = bbox
    
    # Draw the rectangle on the image
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    # Save the image with the drawn bbox
    img.save(output_path)

def draw_and_save_bbox2(frame, bbox, output_path):
    """
    Draws a rectangle (bbox) over an image (numpy array) using OpenCV and saves the resultant image.

    Args:
        frame: A numpy array representing an image.
        bbox: A numpy array with the shape (2, 2) representing the bounding box.
        output_path: Path to save the resultant image.

    Returns:
        None
    """
    # Ensure frame is a numpy array
    assert isinstance(frame, np.ndarray)
    
    # Convert frame to uint8 type if not already
    frame = frame.astype(np.uint8)
    
    # Convert bbox to integer type
    bbox = bbox.astype(int)

    # Convert bbox to tuple format for cv2
    top_left = tuple(bbox[0])
    bottom_right = tuple(bbox[1])

    # Draw the rectangle on the frame. 
    # Here, the color is set to red (0, 0, 255) and thickness to 2.
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
    
    # Save the image with the drawn bbox
    cv2.imwrite(output_path, frame)

def get_files_by_type(folder_path, file_type):
    file_list = []

    for file in os.listdir(folder_path):
        if file.endswith(file_type) and os.path.isfile(os.path.join(folder_path, file)):
            file_list.append(os.path.join(folder_path, file))

    return file_list

def custom_converter(value):
    try:
        return float(value)
    except ValueError:
        return value

def flatten_bbox(bbox):
    return [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]

if __name__ == "__main__":

    colors_dict = {
        "Red": (255, 0, 0),
        "Lime": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "Silver": (192, 192, 192),
        "Gray": (128, 128, 128),
        "Maroon": (128, 0, 0),
        "Olive": (128, 128, 0),
        "Green": (0, 128, 0),
        "Purple": (128, 0, 128),
        "Teal": (0, 128, 128),
        "Navy": (0, 0, 128),
        "Dark Red": (139, 0, 0),
        "Brown": (165, 42, 42),
        "Firebrick": (178, 34, 34),
        "Crimson": (220, 20, 60),
        "Tomato": (255, 99, 71),
        "Coral": (255, 127, 80),
        "Indian Red": (205, 92, 92),
        "Light Coral": (240, 128, 128),
        "Dark Salmon": (233, 150, 122),
        "Salmon": (250, 128, 114),
        "Light Salmon": (255, 160, 122),
        "Orange Red": (255, 69, 0),
        "Dark Orange": (255, 140, 0),
        "Orange": (255, 165, 0),
        "Gold": (255, 215, 0),
        "Dark Golden Rod": (184, 134, 11)
    }

    colors_list = [colors_dict[key] for key in colors_dict]
    colors_name_list = list(colors_dict.keys())


    # ### Set parameters for input and output
    videos_folder = "data/inputs/videos"
    output_folder = "data/outputs/videos/tracking"

    videos_paths = sorted(get_files_by_type(videos_folder, ".avi"))
    # initialize an empty list to store the first 5 characters of each row



    #for video_path in videos_paths:
    empty_bbox = [[0, 0], [0, 0]]
    from tqdm import tqdm
    for video_path in tqdm(videos_paths):
        
        #video_path = os.path.join(videos_folder, video_ID + ".avi")
        video_ID = os.path.basename(video_path).split(".")[0]

        video_name = video_ID
        io_args = {
            'input_video': f'./assets/{video_name}.avi',
            'output_mask_dir': f'./assets/{video_name}_masks', # save pred masks
            'output_video': f'./assets/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
            'output_gif': f'./assets/{video_name}_seg.gif', # mask visualization
            'output_bbox_dir': f'./assets/{video_name}_bbox', # bbox visualization
            'output_bbox_gif': f'./assets/{video_name}_bbox.gif', # bbox visualization

        }

        io_args = {
            'input_video': video_path,
            'output_mask_dir': os.path.join(output_folder, f'{video_name}_masks'), # save pred masks
            'output_video': os.path.join(output_folder, f'{video_name}_seg.mp4'), # mask+frame vizualization, mp4 or avi, else the same as input video
            'output_gif': os.path.join(output_folder, f'{video_name}_seg.gif'), # mask visualization
            'output_bbox_dir': os.path.join(output_folder, f'{video_name}_bbox'), # bbox visualization
            'output_bbox_gif': os.path.join(output_folder, f'{video_name}_bbox.gif'), # bbox visualization
            'output_bbox_csv': os.path.join(output_folder, f'{video_name}.csv'), # bbox csv
            'output_bboxes': os.path.join(output_folder, f'{video_name}.txt'), # bbox csv

        }

        if os.path.exists(io_args['output_video']):
            continue

        print(video_ID)
        print("#"*15)
        print("#"*15)

        # ### Tuning Grounding-DINO and SAM on the First Frame for Good Initialization

        # choose good parameters in sam_args based on the first frame segmentation result
        # other arguments can be modified in model_args.py
        # note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
        sam_args['generator_args'] = {
                'points_per_side': 30,
                'pred_iou_thresh': 0.8,
                'stability_score_thresh': 0.9,
                'crop_n_layers': 1,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 200,
            }

        # Set Text args
        '''
        parameter:
            grounding_caption: Text prompt to detect objects in key-frames
            box_threshold: threshold for box 
            text_threshold: threshold for label(text)
            box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
            reset_image: reset the image embeddings for SAM
        '''
        grounding_caption = "heads"
        grounding_caption = "humans"

        box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True



        # For every sam_gap frames, we use SAM to find new objects and add them for tracking
        # larger sam_gap is faster but may not spot new objects in time
        segtracker_args = {
            'sam_gap': 850, # the interval to run sam to segment new objects
            'min_area': 200, # minimal mask area to add a new mask as a new object
            'max_obj_num': 255, # maximal object number to track in a video
            'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
        }

        # source video to segment
        cap = cv2.VideoCapture(io_args['input_video'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        # output masks
        output_dir = io_args['output_mask_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(io_args['output_bbox_dir']):
            os.makedirs(io_args['output_bbox_dir'])
            
        pred_list = []
        masked_pred_list = []
        bbox_pred_list = []
        bbox_final_list = []
        frames_bbox = []

        torch.cuda.empty_cache()
        gc.collect()
        sam_gap = segtracker_args['sam_gap']
        frame_idx = 0
        segtracker = SegTracker(segtracker_args, sam_args, aot_args)
        segtracker.restart_tracker()

        sampling_freq = 7

        with torch.cuda.amp.autocast():
            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if frame_idx == 0:
                    pred_mask, annotated_frame, boxes = segtracker.detect_and_seg_mod(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                    # pred_mask = cv2.imread('./debug/first_frame_mask.png', 0)
                    init_res = draw_mask(annotated_frame, pred_mask,id_countour=False)
                    torch.cuda.empty_cache()
                    gc.collect()
                    segtracker.add_reference(frame, pred_mask)
                    bbox_final_list.append([frame_idx, boxes[0], 1])

                elif (frame_idx % sam_gap) == 0:
                    seg_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                    save_prediction(seg_mask, './debug/seg_result', str(frame_idx)+'.png')
                    torch.cuda.empty_cache()
                    gc.collect()
                    track_mask = segtracker.track(frame)
                    save_prediction(track_mask, './debug/aot_result', str(frame_idx)+'.png')
                    # find new objects, and update tracker with new objects
                    new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                    if np.sum(new_obj_mask > 0) >  frame.shape[0] * frame.shape[1] * 0.4:
                        new_obj_mask = np.zeros_like(new_obj_mask)
                    save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
                    pred_mask = track_mask + new_obj_mask
                    # segtracker.restart_tracker()
                    segtracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = segtracker.track(frame,update_memory=True)
                    #bbox_mask = get_bbox_from_masked_frame(pred_mask)

                    #print(bbox_mask)
                    boxes, annotated_frame = segtracker.detect_only(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image, bbox_search=[[578, 179], [720, 437]])
                    init_res = draw_mask(annotated_frame, pred_mask,id_countour=False)

                # Create a copy of the original frame
                frame_copy = np.copy(frame)

                # Generate 30 distinct colors using HSV-to-RGB conversion
                colors = []
                for i in range(35):
                    hue = int(255 * (i / 35.0))
                    col_hsv = np.uint8([[[hue, 255, 255]]])  # HSV format
                    col_rgb = cv2.cvtColor(col_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    colors.append(tuple(map(int, col_rgb)))

                # Plot each bbox onto the copied frame with a different color from the list
                for index, bbox in enumerate(boxes):
                    (x1, y1), (x2, y2) = bbox
                    color = colors_list[index]  # Cycle through the color list
                    thickness = 2
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)

                # Save the copied frame with bboxes to a file
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

                path_output_gif = os.path.join(io_args['output_bbox_dir'],str(frame_idx)+'.png')
                cv2.imwrite(path_output_gif, frame_copy)


                torch.cuda.empty_cache()
                gc.collect()
                
                save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')

                
                pred_list.append(pred_mask)
                bbox_pred_list.append(init_res)

                frame_data = {
                    "frame_idx": frame_idx,
                    "boxes": boxes,
                    "colors": colors_name_list[:len(boxes)]
                }
                frames_bbox.append(frame_data)
                
                
                print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
                frame_idx += 1
            cap.release()
            print('\nfinished')
            # Convert bbox_final_list to a list of lists with frame_idx, x0, y0, x1, y1
            bbox_csv_data = [[item[0]] + flatten_bbox(item[1] + [item[-1]] ) for item in bbox_final_list]

            # Define the CSV file path
            csv_file_path = io_args['output_bboxes']
            
            # Open a file in write mode
            with open(csv_file_path, "w") as file:
                # Iterate over each frame
                for frame in frames_bbox:
                    frame_idx = frame["frame_idx"]
                    for box, color in zip(frame["boxes"], frame["colors"]):
                        # Convert the box coordinates to a string
                        box_str = ", ".join(map(str, box))
                        # Write the frame index and box coordinates to the file
                        file.write(f"Frame {frame_idx}: {box_str}, color {color}\n")

        # ### Save results for visualization
        # draw pred mask on frame and save as a video
        cap = cv2.VideoCapture(io_args['input_video'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if io_args['input_video'][-3:]=='mp4':
            fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
        elif io_args['input_video'][-3:] == 'avi':
            fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
            # fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        reduced_width, reduced_height = 640, 360  # Lower resolution

        codec = cv2.VideoWriter_fourcc(*'mp4v')  # Using H.264 codec

        #out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))
        out = cv2.VideoWriter(io_args['output_video'], codec, fps//sampling_freq, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pred_mask = pred_list[frame_idx]
            masked_frame = draw_mask(frame,pred_mask)
            # masked_frame = masked_pred_list[frame_idx]
            masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
            out.write(masked_frame)
            print('frame {} writed'.format(frame_idx),end='\r')
            frame_idx += 1
        out.release()
        cap.release()
        print("\n{} saved".format(io_args['output_video']))
        print('\nfinished')