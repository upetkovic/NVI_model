import os
import glob
import pandas as pd
import json
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

f_dir_path_old = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_dist"
f_dir_path_new = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_dist_emot"

model_name='enet_b0_8_best_vgaf'
emotion_path = "/home/uros/Documents/project31/programming/NI/classifiers/extract_emotions/predictions" + model_name


fer=HSEmotionRecognizer(model_name=model_name)
scores_dic = fer.idx_to_class.copy()

headers = ["face_visibility"]
[headers.append(head) for head in scores_dic.values()]

#json paths
json_paths = glob.glob(os.path.join(emotion_path, "*.json"))

if not os.path.exists(f_dir_path_new):
    os.makedirs(f_dir_path_new)

max_value = 11.577786445617676
min_value = -23.635536193847656
max_value = -min_value 
def normalize01(value, min_value, max_value):
    interval_l = max_value - min_value
    k = 1 / interval_l
    n = -k*min_value
    return value * k + n

for json_path in json_paths:
    videoID = os.path.basename(json_path).split(".")[0]
    csv_old_path = os.path.join(f_dir_path_old, videoID + ".csv")
    csv_new_path = os.path.join(f_dir_path_new, videoID + ".csv")

    with open(json_path, 'r') as file:
        emotions = json.load(file)

    df = pd.read_csv(csv_old_path)
    N_frames = df['frame_id'].iloc[-1]
    new_columns = {}
    for header in headers:
        new_columns[header] = []

    for frame_id in df['frame_id']:

        for ind, header in enumerate(headers):
            frame_values = emotions[str(int(frame_id))]

            if frame_values['emotion'] == "Unknown":
                if ind == 0:
                    new_columns[header].append(0)
                else:
                    new_columns[header].append(0.5)
            else:
                if ind == 0:
                    new_columns[header].append(1)
                else:
                    new_columns[header].append(normalize01(frame_values['scores'][ind - 1], min_value, max_value))
    
    for header, values in new_columns.items():
        #print(values)
        df[header] = values
    df.to_csv(csv_new_path, index=False)
    




