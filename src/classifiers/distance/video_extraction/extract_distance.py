import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
import torch.nn.functional as F
import sys
sys.path.append("/home/uros/Documents/project31/programming/NI/classifiers/distance/model")
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from model import FinetuneResnet
import csv
from PIL import Image


DEPTH_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/depth/real_depth7"
SEGM_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/semantic_segmentation/humans/real_segm7"
FEATURES_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/uros_scioi/tubCloud/project31/data/Talis_videos/features_all"
OUTPUT_FOLDER = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI_test/distance/results"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI_classifiers/distance/weights/loss001154corr0552.pth"
model = FinetuneResnet()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.float()



def preprocess_image(image, scale_factor = 0.33):
    H = int(image.shape[1] * scale_factor)
    W = int(image.shape[2] * scale_factor)
    image = image.unsqueeze(0)
    image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)
    image = image.squeeze(0)
    return image

# import model

# load weights

# get all paths to feature
file_list = [f for f in os.listdir(DEPTH_FOLDER) if os.path.isfile(os.path.join(DEPTH_FOLDER, f))]
file_list.sort()
head_headers = ['keypoint1X', 'keypoint1Y', 'keypoint2X', 'keypoint2Y', 'keypoint3X', 'keypoint3Y', 'keypoint4X', 'keypoint4Y', 'keypoint5X', 'keypoint5Y']
# loop over paths
for file_id in file_list:
    print(file_id)
    file_id = "00281.npy"
    csv_id = file_id.split(".")[0] + ".csv"
    depth_path = os.path.join(DEPTH_FOLDER, file_id)
    segm_path = os.path.join(SEGM_FOLDER, file_id)
    csv_path = os.path.join(FEATURES_FOLDER, csv_id)
    # load mask and segm

    depth_images = np.load(depth_path)
    segm_images = np.load(segm_path)

    features = pd.read_csv(csv_path)
    headers = features.columns.tolist()
    features_by_frame = features.set_index('frame_id').T.to_dict('dict')
    
    #loop over frames
    unknown_frames_list = []
    batch_size = int(np.ceil(len(features_by_frame.keys()) / 7.0))
    image_batch = torch.zeros(size=(batch_size, 3, 237, 422 ))
    batch_idx = 0
    frame_ids_list = []
    for frame_id in features_by_frame.keys():
        frame_id = int(frame_id)
        if frame_id % 7 == 0:
            frame_ids_list.append(frame_id)
            depth_image = depth_images[int(frame_id/7)]
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

            if teacher_ID==0:
                unknown_frames_list.append(frame_id)

            teacher_mask = (segm_image == teacher_ID).astype(int)
            student_mask = ((segm_image != 0) & (segm_image != teacher_ID)).astype(int)

            input_image = np.zeros(shape=(3, depth_image.shape[0], depth_image.shape[1]))
            input_image[2, :, :] = depth_image / 6 
            input_image[0, :, :] = teacher_mask
            input_image[1, :, :] = student_mask
            image = torch.from_numpy(input_image)
            image = preprocess_image(image.float())
            image_batch[batch_idx, :, :, :] = image
            
            
            
            
            img = image.detach().cpu().numpy()  # shape: (3, W, H)

            img = np.transpose(img, (1, 2, 0))
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

            Image.fromarray(img).save("output_path2.png")

            #exit()

            batch_idx += 1
    
    model.eval()
    
    with torch.no_grad():        
        results = model(image_batch.to(device)).to('cpu').numpy()

    print(results)
    #exit()
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

    csv_path = os.path.join(OUTPUT_FOLDER, csv_id)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "distance"])  # Column headers
        writer.writerows(zip(list(frames_out), list(results_out)))
    

    exit()




# load tracking info

# make transformations

