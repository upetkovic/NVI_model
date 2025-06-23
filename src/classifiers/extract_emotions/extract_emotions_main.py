import os
import cv2
from external.hsemotion.hsemotion.facial_emotions import HSEmotionRecognizer
import glob, os
from tqdm import tqdm 
import json, csv
from src.utils.video_utils import get_images_paths

model_name='enet_b0_8_best_vgaf'
fer=HSEmotionRecognizer(model_name=model_name)
scores_dic = fer.idx_to_class.copy()
cropped_face_path = "/home/uros/Documents/project31/programming/NI/face_extraction/cropped_faces"
cropped_face_path = "/home/uros/Documents/project31/programming/NI/face_extraction/cropped_faces_part22"

output_path = "./predictions_part2" + model_name



# Here we wrap the outer loop with tqdm to create a progress bar for video folders processing.
def extract_emotions(cropped_face_path, output_json, N_frames):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    video_id = os.path.basename(cropped_face_path)
    output_csv = os.path.join(os.path.dirname(output_json), video_id + ".csv")

    output_dic = {}

    for i in range(0, N_frames):
        output_dic[i] = {'emotion': "Unknown", 'scores':[]}

    # Now, we also wrap the inner loop with tqdm to create a progress bar for frames processing within each video.
    # Note: We use a new tqdm instance here with a different description to distinguish it from the outer loop.
    for frame_path in get_images_paths(cropped_face_path):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped_face_path", required=True, help="Path to json FOLDER.")
    parser.add_argument("--output_json", required=True, help="Path to tracking folder.")
    parser.add_argument("--N_frames", required=True, help="Path to tracking folder.")
    #args = parser.parse_args()

    extract_emotions(
        cropped_face_path="data/outputs/teacher_faces/00317",
        output_json="data/outputs/emotions/00317.json",
        N_frames=108
        )

"""
    extract_emotions(
        cropped_face_path=args.cropped_face_path,
        output_json=args.output_json,
        N_frames=args.N_frames,
    )

"""