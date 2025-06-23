import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from torchvision import transforms

# Create a simple custom Dataset
class DistanceDataset(Dataset):
    def __init__(self, depth_path, segm_path, ratings_path, std_th=500):
        self.depth_path = depth_path
        self.segm_path = segm_path
        self.ratings_by_id = pd.read_csv(ratings_path).set_index('imgID').T.to_dict('dict')
        self.std_th = std_th
        self.mean_output = 0

        self.keys_to_remove = []
        for key in self.ratings_by_id.keys():
            ratings = torch.tensor([self.ratings_by_id[key]['distance0'], self.ratings_by_id[key]['distance1'], self.ratings_by_id[key]['distance2']])
            std = torch.std(ratings)
            self.ratings_by_id[key]['ratings'] = ratings

            if std < self.std_th:
                npy_id = key.split('.')[0] + ".npy"
                # load depth
                depth_image = np.load(os.path.join(self.depth_path, npy_id))
                input_image = np.zeros(shape=(3, depth_image.shape[0], depth_image.shape[1]))
                input_image[2, :, :] = depth_image / 6 ##normalize

                segm_image = np.load(os.path.join(self.segm_path, npy_id))


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
                    input_image[0, :, :] = teacher_mask
                    #segm_image_npy[0, :, :] = teacher_mask*segm_image_npy[0, :, :]

                    # student channel             
                    student_mask = ((segm_image != 0) & (segm_image != teacher_ID)).astype(int)
                    input_image[1, :, :] = student_mask
                    #segm_image_npy[1, :, :] = student_mask*segm_image_npy[1, :, :]

                    self.ratings_by_id[key]['input_image'] = torch.from_numpy(input_image)
                    ### transform images & downsample image
                    self.ratings_by_id[key]['input_image'] = self.preprocess_image(self.ratings_by_id[key]['input_image'])

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


    def preprocess_image(self, image, scale_factor = 0.33):
        H = int(image.shape[1] * scale_factor)
        W = int(image.shape[2] * scale_factor)
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)
        image = image.squeeze(0)

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
    dataset_val = DistanceDataset(depth_path=depth_path, segm_path=segm_path, ratings_path=ratings_path)

