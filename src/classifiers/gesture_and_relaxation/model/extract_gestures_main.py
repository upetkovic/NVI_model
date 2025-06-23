from classifiers.gesture_and_relaxation.model.model import FinetuneResnet
import torch
import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import cv2
import csv
from PIL import Image
import torchvision.utils as vutils

DEPTH_FOLDER = "./data/outputs/depth"
SEGM_FOLDER = "./data/outputs/tracking"

FEATURES_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/uros_scioi/tubCloud/project31/data/Talis_videos/features_all_dist"
OUTPUT_FOLDER = "data/outputs/gesture"



def find_bounding_box(image):
    # Convert RGB image to grayscale by averaging across channels
    gray_image = torch.mean(image.float(), dim=0)

    # Considering all non-zero values as part of the mask
    rows = torch.any(gray_image > 0, dim=1)
    cols = torch.any(gray_image > 0, dim=0)
    try:
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
    except:
        xmin = 0
        ymin = 0
        xmax = 2
        ymax = 2
    return xmin, ymin, xmax, ymax

def resize_with_padding(img, desired_size):
    img = img.float()
    target_ratio = desired_size[0] / desired_size[1]
    src_ratio = img.shape[2] / img.shape[1]  # Changed to consider channels

    if src_ratio > target_ratio:
        new_width = desired_size[0]
        new_height = int(desired_size[0] / src_ratio)
    else:
        new_width = int(desired_size[1] * src_ratio)
        new_height = desired_size[1]

    # Resize while preserving the ratio
    resized_img = F.interpolate(img.unsqueeze(0), (new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

    # Padding
    padding_left = (desired_size[0] - new_width) // 2
    padding_right = desired_size[0] - new_width - padding_left
    padding_top = (desired_size[1] - new_height) // 2
    padding_bottom = desired_size[1] - new_height - padding_top
    padded_img = F.pad(resized_img, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)
    return padded_img


def crop_image(img, desired_size):
    xmin, ymin, xmax, ymax = find_bounding_box(img)
    cropped_img = img[:, ymin:ymax+1, xmin:xmax+1]  # Added channel slicing
    resized_padded_img = resize_with_padding(cropped_img, desired_size)
    return resized_padded_img


def preprocess_image(frame_img, teacher_mask):
    expanded_mask = np.expand_dims(teacher_mask, axis=-1)

    three_channel_mask = np.repeat(expanded_mask, 3, axis=-1)
    three_channel_mask = torch.tensor(three_channel_mask).permute(2, 0, 1)
    masked_frame = frame_img * three_channel_mask
    #self.ratings_by_id[key]['input_image'] = masked_frame.unsqueeze(0)
    masked_frame = torch.tensor(masked_frame)
    input_image = crop_image(masked_frame, (360, 360))
    input_image = input_image.unsqueeze(0)
    #vutils.save_image(self.ratings_by_id[key]['input_image'], 'saved_image.jpg')
    #vutils.save_image(masked_frame, 'saved_image_full.jpg')


    ### transform images & downsample image
    #normalize
    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    input_image = normalize_img(input_image).squeeze()
    return input_image


def extract_gestures(segm_folder_teacher_path, csv_path_output, video_path):
    

    weights_path = "checkpoints/gesture_loss002807corr0838acc849.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FinetuneResnet()
    model = model.float()
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    #file_id = os.path.basename(video_path).split(".")[0]
    #csv_id = file_id + ".csv"
    #print(file_id)

    #segm_path = os.path.join(segm_folder, file_id + ".npy")
    #csv_path_output = os.path.join(output_folder, csv_id)

    #segm_path = os.path.join(segm_folder, file_id)
    #segm_folder_path = os.path.join(segm_folder, file_id.split(".")[0] + "_masks")
    #segm_folder_teacher_path = os.path.join(segm_folder, file_id.split(".")[0] + "_teacher")

    mask_list = [f for f in os.listdir(segm_folder_teacher_path) if os.path.isfile(os.path.join(segm_folder_teacher_path, f))]
    mask_list.sort()
    masks_all = []
    [masks_all.append(int(mask.split(".")[0])) for mask in mask_list]
    num_frames = max(masks_all) + 1

    # load mask and segm
    ## load all frames
    frames = {}
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    frame_cnt = 0
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
    else:
        # Loop through each frame in the video
        while True:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                break
            if frame_cnt % 1 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_tensor = torch.tensor(rgb_frame).permute(2, 0, 1) / 255.0

                frames[frame_cnt] = frame_tensor

            frame_cnt += 1


    #loop over frames
    #batch_size = int(np.ceil() / 7.0))
    batch_size = len(masks_all)
    batch_size = 108

    image_batch = torch.zeros(size=(batch_size, 3, 360, 360))
    batch_idx = 0
    frame_ids_list = []
    results_batches = []

    # Loop over each batch
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        current_batch_size = batch_end - batch_start

        image_batch = torch.zeros(size=(current_batch_size, 3, 360, 360))
        frame_ids_list = []

        # Process each frame in the current batch
        for batch_idx, frame_id in enumerate(range(batch_start, batch_end)):
            frame_id = int(frame_id)
            frame_ids_list.append(frame_id)

            # Process segmentation teacher image
            segm_teacher_path = os.path.join(segm_folder_teacher_path, f"{frame_id:05d}" + ".png")
            with Image.open(segm_teacher_path) as img:
                grayscale_img = img.convert('L')
                teacher_mask = np.array(grayscale_img)
                teacher_mask = (teacher_mask > 0).astype(int)

            """
            # Process segmentation student image
            segm_path = os.path.join(segm_folder_path, f"{frame_id:05d}" + ".png")
            with Image.open(segm_path) as img:
                grayscale_img = img.convert('L')
                student_mask = np.array(grayscale_img)
                student_mask = (student_mask > 0).astype(int)
                student_mask = student_mask - teacher_mask
            """
            # Prepare input image
            # Assuming preprocess_image function can handle frames and masks
            image = preprocess_image(frames[frame_id], teacher_mask)
            image_batch[batch_idx, :, :, :] = image

        # Apply model to the current batch
        model.eval()
        with torch.no_grad():
            results = model(image_batch.to(device)).to('cpu').numpy()
            [results_batches.append(result) for result in results]



    frames_out = list(range(num_frames))
    results_out = np.array(results_batches).squeeze().tolist()

    with open(csv_path_output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "gesture_cont"])  # Column headers
        writer.writerows(zip(list(frames_out), list(results_out)))


if __name__ == "__main__":

    """
    extract_gestures(
        segm_folder_teacher_path="data/outputs/tracking/00281_teacher",
        csv_path_output="data/outputs/gesture/00281.csv",
        video_path="./data/inputs/videos/00281.avi")
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_segm", required=True, help="Path to the folder with teacher's masks")
    parser.add_argument("--output_path", required=True, help="Path to the output csv")
    parser.add_argument("--video_path", required=True, help="path to the video")
    args = parser.parse_args()

    extract_gestures(
        segm_folder_teacher_path=args.teacher_segm,
        csv_path_output=args.output_path,
        video_path=args.video_path
    )
