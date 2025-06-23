import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import torchvision.utils as vutils

# Create a simple custom Dataset
class GestureDataset(Dataset):
    def __init__(self, depth_path, segm_path, ratings_path, frame_path, std_th=500):
        self.depth_path = depth_path
        self.frame_path = frame_path
        self.segm_path = segm_path
        self.ratings_by_id = pd.read_csv(ratings_path).set_index('imgID').T.to_dict('dict')
        self.std_th = std_th
        self.mean_output = 0

        self.keys_to_remove = []
        for key in self.ratings_by_id.keys():
            ratings = torch.tensor([self.ratings_by_id[key]['distance0'], self.ratings_by_id[key]['distance1'], self.ratings_by_id[key]['distance2']])
            ratings = torch.tensor([self.ratings_by_id[key]['gesture0'], self.ratings_by_id[key]['gesture1'], self.ratings_by_id[key]['gesture2']])
            #ratings = torch.tensor([self.ratings_by_id[key]['relaxation0'], self.ratings_by_id[key]['relaxation1'], self.ratings_by_id[key]['relaxation2']])
            rater_id = 0
            behaviour = 'gesture'
            {behaviour}
            #ratings = torch.tensor([self.ratings_by_id[key][f'relaxation{rater_id}'], self.ratings_by_id[key][f'relaxation{rater_id}'], self.ratings_by_id[key][f'relaxation{rater_id}']])
            ratings = torch.tensor([self.ratings_by_id[key][f'{behaviour}{rater_id}'], self.ratings_by_id[key][f'{behaviour}{rater_id}'], self.ratings_by_id[key][f'{behaviour}{rater_id}']])
            ratings = torch.tensor([self.ratings_by_id[key][f'{behaviour}0'], self.ratings_by_id[key][f'{behaviour}1'], self.ratings_by_id[key][f'{behaviour}2']])

            std = torch.std(ratings)
            self.ratings_by_id[key]['ratings'] = ratings

            if std < self.std_th:
                npy_id = key.split('.')[0] + ".npy"
                png_id = key.split('.')[0] + ".png"
                # load depth
                #depth_image = np.load(os.path.join(self.depth_path, npy_id))
                #input_image = np.zeros(shape=(3, depth_image.shape[0], depth_image.shape[1]))
                #input_image[2, :, :] = depth_image / 6 ##normalize

                segm_image = np.load(os.path.join(self.segm_path, npy_id))

                # load grayscale image
                gray_img = Image.open(os.path.join(self.frame_path, png_id))
                #gray_img = Image.open(os.path.join(self.frame_path, png_id)).convert('L')

                frame_img = ToTensor()(gray_img).squeeze()


                # get teacher ID
                posX = int(self.ratings_by_id[key]['posX'])
                posY = int(self.ratings_by_id[key]['posY'])
                teacher_ID = segm_image[720 - posY, posX]

                
                # get all student IDs
                unique_ids = np.unique(segm_image)
                studentIDs = unique_ids[(unique_ids != 0) & (unique_ids != teacher_ID)]
                n_students = len(studentIDs)

                if teacher_ID != 0:
                    # teacher channel
                    teacher_mask = (segm_image == teacher_ID).astype(int)
                    #input_image[0, :, :] = teacher_mask
                    #segm_image_npy[0, :, :] = teacher_mask*segm_image_npy[0, :, :]

                    # student channel             
                    student_mask = ((segm_image != 0) & (segm_image != teacher_ID)).astype(int)
                    #input_image[1, :, :] = student_mask
                    #segm_image_npy[1, :, :] = student_mask*segm_image_npy[1, :, :]

                    #self.ratings_by_id[key]['input_image'] = torch.from_numpy(input_image)
                    masked_frame = frame_img * teacher_mask
                    #self.ratings_by_id[key]['input_image'] = masked_frame.unsqueeze(0)

                    self.ratings_by_id[key]['input_image'] = self.crop_image(masked_frame, (360, 360))
                    self.ratings_by_id[key]['input_image'] = self.ratings_by_id[key]['input_image'].unsqueeze(0)
                    #vutils.save_image(self.ratings_by_id[key]['input_image'], 'saved_image.jpg')
                    #vutils.save_image(masked_frame, 'saved_image_full.jpg')


                    ### transform images & downsample image
                    #normalize
                    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                    self.ratings_by_id[key]['input_image'] = normalize_img(self.ratings_by_id[key]['input_image']).squeeze()

                else:
                    self.keys_to_remove.append(key)

            else:
                #remove the key
                self.keys_to_remove.append(key)

        for key in self.keys_to_remove:
            del self.ratings_by_id[key]

        self.N_data = len(self.ratings_by_id.keys())
        ## enumerate dictionary
        self.dict_idx = dict(zip(list(range(self.N_data)), self.ratings_by_id.keys()))
        # remove frames with high variance between raters
        self.compute_mean_output()

    def compute_mean_output(self):
        all_outputs = np.zeros(self.N_data)
        for i in range(self.N_data):
            imgID = self.dict_idx[i]
            all_outputs[i] = torch.median(self.ratings_by_id[imgID]['ratings'] / 10000)
        
        self.mean_output = np.mean(all_outputs)
        return 1

    def find_bounding_box(self, image):
        # Convert RGB image to grayscale by averaging across channels
        gray_image = torch.mean(image, dim=0)

        # Considering all non-zero values as part of the mask
        rows = torch.any(gray_image > 0, dim=1)
        cols = torch.any(gray_image > 0, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        return xmin, ymin, xmax, ymax

    def resize_with_padding(self, img, desired_size):
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

    def crop_image(self, img, desired_size):
        xmin, ymin, xmax, ymax = self.find_bounding_box(img)
        cropped_img = img[:, ymin:ymax+1, xmin:xmax+1]  # Added channel slicing
        resized_padded_img = self.resize_with_padding(cropped_img, desired_size)
        return resized_padded_img
    
    def preprocess_image(self, image, scale_factor = 0.33):
        H = int(image.shape[1] * scale_factor)
        W = int(image.shape[2] * scale_factor)
        #image = image.unsqueeze(0)
        image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)
        image = image.squeeze(0).squeeze(0)

        return image
    def __len__(self):
        return self.N_data 

    def __getitem__(self, idx):
        imgID = self.dict_idx[idx]
        labels = torch.median(self.ratings_by_id[imgID]['ratings'] / 10000)
        image = self.ratings_by_id[imgID]['input_image']

        self.horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        image = self.horizontal_flip(image)
        return image.float(), labels, imgID


if __name__ == "__main__":
    ratings_path = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL.csv"
    depth_path = "/home/uros/Documents/project31/data/classifiers_frames/depth_frames/real_depth"
    segm_path = "/home/uros/Documents/project31/data/classifiers_frames/segm_frames_humans/real_segm"
    frame_path = "/home/uros/Documents/project31/data/classifiers_frames/Talis_frames15_v2"
    dataset_val = GestureDataset(depth_path=depth_path, segm_path=segm_path, frame_path=frame_path, ratings_path=ratings_path)

