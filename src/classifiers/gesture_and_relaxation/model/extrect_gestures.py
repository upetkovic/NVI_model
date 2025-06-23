from model import FinetuneResnet
import torch
import os
import numpy as np
import glob
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
from collections import Counter
import cv2
from scipy.interpolate import interp1d
import csv

loss_dir = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI_classifiers/gesture/weights"
weights_path =  "loss002807corr0838acc849.pth"
weights_path = os.path.join(loss_dir, weights_path)

segm_dir = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/semantic_segmentation/humans/real_segm7"
videos_dir = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/TalisTeacherVideos"
videos_path = glob.glob(os.path.join(videos_dir, "*.avi"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FinetuneResnet()
model = model.float()
model.load_state_dict(torch.load(weights_path))
model.to(device)





DEPTH_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/depth/real_depth7"
SEGM_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/semantic_segmentation/humans/real_segm7"
FEATURES_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/uros_scioi/tubCloud/project31/data/Talis_videos/features_all_dist"
OUTPUT_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/gesture/results"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_bounding_box(image):
    # Convert RGB image to grayscale by averaging across channels
    gray_image = torch.mean(image.float(), dim=0)

    # Considering all non-zero values as part of the mask
    rows = torch.any(gray_image > 0, dim=1)
    cols = torch.any(gray_image > 0, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
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


# import model

# load weights

# get all paths to feature
file_list = [f for f in os.listdir(DEPTH_FOLDER) if os.path.isfile(os.path.join(DEPTH_FOLDER, f))]
file_list.sort()
head_headers = ['keypoint1X', 'keypoint1Y', 'keypoint2X', 'keypoint2Y', 'keypoint3X', 'keypoint3Y', 'keypoint4X', 'keypoint4Y', 'keypoint5X', 'keypoint5Y']
# loop over paths
for video_path in videos_path:
    file_id = os.path.basename(video_path).split(".")[0]
    csv_id = file_id + ".csv"
    print(file_id)
    depth_path = os.path.join(DEPTH_FOLDER, file_id)

    segm_path = os.path.join(SEGM_FOLDER, file_id + ".npy")
    csv_path = os.path.join(FEATURES_FOLDER, csv_id)
    csv_path_output = os.path.join(OUTPUT_FOLDER, csv_id)

    if os.path.exists(csv_path_output):
        continue
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
            if frame_cnt % 7 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_tensor = torch.tensor(rgb_frame).permute(2, 0, 1) / 255.0

                frames[frame_cnt] = frame_tensor

            frame_cnt += 1

            # Break the loop if the video is over (ret will be False if no frame is read)
            if not ret:
                break
    #depth_images = np.load(depth_path)
    segm_images = np.load(segm_path)

    features = pd.read_csv(csv_path)
    headers = features.columns.tolist()
    features_by_frame = features.set_index('frame_id').T.to_dict('dict')
    
    #loop over frames
    unknown_frames_list = []
    batch_size = int(np.ceil(len(features_by_frame.keys()) / 7.0))
    image_batch = torch.zeros(size=(batch_size, 3, 360, 360))
    batch_idx = 0
    frame_ids_list = []
    for frame_id in features_by_frame.keys():
        frame_id = int(frame_id)
        if frame_id % 7 == 0:
            frame_ids_list.append(frame_id)
            segm_image = segm_images[int(frame_id/7)]
            head_keypoints = np.zeros(shape=(5, 2))
            for i in range(5):
                if features_by_frame[frame_id][head_headers[2*i]] == 'Unknown':
                    unknown_frames_list.append(frame_id)
                    pass
                else:    
                    head_keypoints[i, 0] = int(float(features_by_frame[frame_id][head_headers[2*i]]))
                    head_keypoints[i, 1] = int(float(features_by_frame[frame_id][head_headers[2*i + 1]]))

            teacher_ids = []
            for i in range(head_keypoints.shape[0]):
                posX = int(head_keypoints[i, 1])
                posY = int(head_keypoints[i, 0])
                if (posX + posY == 0) or posY > 1279 or posX > 719:
                    pass
                else:
                    teacher_ID = segm_image[posX, posY]
                
                    if teacher_ID != 0:
                        teacher_ids.append(teacher_ID)

            if len(teacher_ids) > 0:
                counter = Counter(teacher_ids)
                teacher_ID = counter.most_common(1)[0][0]
            else:
                teacher_ID = 0

            if teacher_ID==0:
                unknown_frames_list.append(frame_id)

            teacher_mask = (segm_image == teacher_ID).astype(int)
            student_mask = ((segm_image != 0) & (segm_image != teacher_ID)).astype(int)

            input_image = np.zeros(shape=(3, 360, 360))
            '''
            input_image[2, :, :] = depth_image / 6 
            input_image[0, :, :] = teacher_mask
            input_image[1, :, :] = student_mask
            image = torch.from_numpy(input_image)
            image = preprocess_image(image.float())
            '''
            image = preprocess_image(frames[frame_id], teacher_mask)
            image_batch[batch_idx, :, :, :] = image
            batch_idx += 1
    with torch.no_grad():        
        results = model(image_batch.to(device)).to('cpu').numpy()

    unknown_frames_list = list(set(unknown_frames_list))
    frame_known = list(set(frame_ids_list) - set(unknown_frames_list)) #get known frames
    
    frame_known.sort()
    results_known = []
    for i in frame_known:
        results_known.append(results[int(i/7)])
    frame_known = np.array(frame_known)
    results_known = np.array(results_known)

    frames_out = list(features_by_frame.keys())

    f = interp1d(frame_known.T, results_known.squeeze(), kind='linear',fill_value="extrapolate")
    results_out = f(frames_out)
    print('#####'*20)
    print(len(results_known))
    for ind, value in enumerate(frames_out):
        frames_out[ind] = int(value)

    with open(csv_path_output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "gesture_cont"])  # Column headers
        writer.writerows(zip(list(frames_out), list(results_out)))
    






# load tracking info

# make transformations

