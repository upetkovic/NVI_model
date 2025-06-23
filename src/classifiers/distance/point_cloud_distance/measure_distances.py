import pandas as pd
import numpy as np
import os
from helpers import depth_to_point_cloud, visualize_point_clouds, compute_centroids, normalize_array
import csv

DEPTH_IMAGE_FOLDER = "/home/uros/Documents/project31/data/classifiers_frames/depth_frames/real_depth"
SEGMENTATED_IMAGE_FOLDER = "/home/uros/Documents/project31/data/classifiers_frames/segm_frames_humans/real_segm"
human_raters_path = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL_and_TRAIN.csv"

labels = pd.read_csv(human_raters_path)
labels_dict = labels.set_index('imgID').T.to_dict('dict')

fraction = 0.95

headers = ['imgID', 'distance0', 'distance1','distance2', f'distance0_f{fraction}',f'distance1_f{fraction}', f'distance2_f{fraction}']
filename = 'distances.csv'

# Writing to CSV
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # write the headers first


n_wanted_distances = 3
# loop over all images
all_distances = np.ones((len(labels['imgID'].to_list()), n_wanted_distances + 3))*(-1)
for img_ind, imgID in enumerate(labels['imgID'].to_list()):
    print(img_ind, imgID)
    imageID = imgID.split(".")[0]
    
    # read depth image
    depth_image = np.load(os.path.join(DEPTH_IMAGE_FOLDER, imageID + ".npy"))

    # read segmentated image
    segm_image = np.load(os.path.join(SEGMENTATED_IMAGE_FOLDER, imageID + ".npy"))

    # get teacher ID
    posX = int(labels_dict[imgID]['posX'])
    posY = int(labels_dict[imgID]['posY'])
    teacher_ID = segm_image[720 - posY, posX]

    # get all student IDs
    unique_ids = np.unique(segm_image)
    studentIDs = unique_ids[(unique_ids != 0) & (unique_ids != teacher_ID)]
    n_students = len(studentIDs)

    if teacher_ID != 0 and n_students > 0:
        teacher_mask = (segm_image == teacher_ID).astype(int)
        teacher_cloud, _ = depth_to_point_cloud(depth_image*teacher_mask)
        

        # for each student measure the distance
        distances_list = []
        for studentID in studentIDs:
            # get mask
            student_mask = (segm_image == studentID).astype(int)
            # transform it to point cloud
            student_cloud, _ = depth_to_point_cloud(depth_image*student_mask)
            #visualize_point_clouds(teacher_cloud, student_cloud)

            centroid_teacher, centroid_student = compute_centroids(teacher_cloud, student_cloud, fraction=fraction, method='mean')

            distance = np.linalg.norm(centroid_teacher - centroid_student)
            distances_list.append(distance)
        # write distance for top 5 students
        distances_sorted = np.sort(distances_list)
        distances_out = np.ones(n_wanted_distances) * (-1)
        for ind, d in enumerate(distances_sorted):
            if ind == n_wanted_distances:
                break 
            distances_out[ind] = d
    
    a = 3

    all_distances[img_ind, 0] = labels_dict[imgID]['distance0']
    all_distances[img_ind, 1] = labels_dict[imgID]['distance1']
    all_distances[img_ind, 2] = labels_dict[imgID]['distance2']

    for i in range(n_wanted_distances):
        all_distances[img_ind, 3 + i] = distances_out[i]
    
    all_distances_filtered = all_distances[~np.any(all_distances == -1, axis=1)]
    distance_ref = np.median(all_distances_filtered[:, :3], axis=1)
    distance_est = all_distances_filtered[:, 3]

    corr_matrix = np.corrcoef(distance_ref, distance_est)
    print(corr_matrix)

    #write to csv
    row = [imgID]
    for i in range(3):
        row.append(all_distances[img_ind, i])
    [row.append(dist) for dist in distances_out]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)  # write each row

distance_ref = np.median(all_distances[:, :3], axis=1)
distance_est = np.median(all_distances[:, 3:4], axis=1)
corr_matrix = np.corrcoef(normalize_array(distance_ref), normalize_array(distance_est))
print(corr_matrix)
