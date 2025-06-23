import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import glob, os
from tqdm import tqdm  # <-- Make sure to import tqdm
import json, csv

model_name='enet_b0_8_best_vgaf'
#model_name='enet_b0_8_va_mtl'
#model_name='enet_b2_8'

fer=HSEmotionRecognizer(model_name=model_name)
scores_dic = fer.idx_to_class.copy()
cropped_face_path = "/home/uros/Documents/project31/programming/NI/face_extraction/cropped_faces"
cropped_face_path = "/home/uros/Documents/project31/programming/NI/face_extraction/cropped_faces_part22"

output_path = "./predictions_part2" + model_name

if not os.path.exists(output_path):
    os.makedirs(output_path)



# get all video_paths
folders_path = glob.glob(os.path.join(cropped_face_path, "*"))
folders_path.sort()
# Here we wrap the outer loop with tqdm to create a progress bar for video folders processing.
for video_path in tqdm(folders_path, desc="Processing videos", unit="video"):
    video_id = os.path.basename(video_path)
    output_json = os.path.join(output_path, video_id + ".json")
    output_csv = os.path.join(output_path, video_id + ".csv")

    
    frames_path = glob.glob(os.path.join(video_path, "*.png"))
    frames_path.sort()
    
    output_dic = {}

    for i in range(0, 752):
        output_dic[i] = {'emotion': "Unknown", 'scores':[]}

    # Now, we also wrap the inner loop with tqdm to create a progress bar for frames processing within each video.
    # Note: We use a new tqdm instance here with a different description to distinguish it from the outer loop.
    for frame_path in tqdm(frames_path, desc=f"Processing frames for video {video_id}", unit="frame"):
        frame_id = os.path.basename(frame_path).split(".")[0]
        face_bgr = cv2.imread(frame_path)
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        emotion, scores = fer.predict_emotions(face, logits=True)

        output_dic[int(frame_id)] = {'emotion': emotion, 'scores':scores.tolist()}
    
    with open(output_json, 'w') as json_file:
        json.dump(output_dic, json_file, indent=4)  # The indent parameter is the number of spaces used for indentation
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Prepare the header based on the emotion classes
        header = ['frame_id', 'emotion'] + [scores_dic[idx] for idx in sorted(scores_dic)]
        
        # Write the header to the CSV file
        writer.writerow(header)

        # Write the data to the CSV file
        for frame_id, value in output_dic.items():
            # Frame id and emotion are the first two columns
            row = [frame_id, value['emotion']]
            
            # Then append each score as a new field in the row
            row.extend(value['scores'])
            
            # Write this row in the CSV file
            writer.writerow(row)