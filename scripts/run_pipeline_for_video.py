import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--video_path", required=True)
args = parser.parse_args()
VIDEO_INPUT_PATH = args.video_path



import os
import subprocess
from utils.helpers import create_command, get_first_json_file

#from src.utils.helpers import create_command
from utils.video_utils import resample_video, resample_with_ffmpeg
from utils.video_utils import get_video_length, get_video_frame_count

from utils.config_paths import json_first_frame_labels_path, OUTPUTS_MAIN_FOLDER
#from src.utils.video_utils import resample_video, resample_with_ffmpeg

## video id
#VIDEO_INPUT_PATH = "./data/inputs/videos/00281.avi"
#OUTPUTS_MAIN_FOLDER = "./data/outputs"

RESAMPLED_VIDEOS_FOLDER = os.path.join(OUTPUTS_MAIN_FOLDER, "resampled_videos")
TRACKING_FOLDER = os.path.join(OUTPUTS_MAIN_FOLDER, "tracking")
DEPTH_FOLDER = os.path.join(OUTPUTS_MAIN_FOLDER, "depth")
DISTANCE_FOLDER = os.path.join(OUTPUTS_MAIN_FOLDER, "distance")
DISTANCE_FOLDER_POST = os.path.join(OUTPUTS_MAIN_FOLDER, "distance_postprocessed")

GESTURE_FOLDER = os.path.join(OUTPUTS_MAIN_FOLDER, "gesture")
GESTURE_FOLDER_POST = os.path.join(OUTPUTS_MAIN_FOLDER, "gesture_postprocessed")



EMOTIONS_FOLDER = os.path.join(OUTPUTS_MAIN_FOLDER, "emotions")
EMOTIONS_FOLDER_FILT = os.path.join(OUTPUTS_MAIN_FOLDER, "emotions_filtered")
EMOTIONS_FOLDER_POST = os.path.join(OUTPUTS_MAIN_FOLDER, "emotions_postprocessed")


FACES_PATH = os.path.join(OUTPUTS_MAIN_FOLDER, "teacher_faces")
FEATURES_FOLDER = os.path.join(OUTPUTS_MAIN_FOLDER, "features")
FEATURES_POST = os.path.join(OUTPUTS_MAIN_FOLDER, "features_postprocessed")

NVI_PREDICTIONS_CSV = os.path.join(OUTPUTS_MAIN_FOLDER, "nvi_predictions.csv")
# VIDEO_INPUT_PATH = os.path.join(OUTPUTS_MAIN_FOLDER, "videos


conda_envs = {
    "nvi": "nvi_env2",
    "dinov2": "dinov22",
    "hsemotion": "hsemotion2"
}

NORMALIZATION_JSON_PATH = "data/config/normalization_parameters.json"
#### PREPARE FOLDERS ####
os.makedirs(RESAMPLED_VIDEOS_FOLDER,exist_ok=True)
os.makedirs(TRACKING_FOLDER,exist_ok=True)
os.makedirs(DEPTH_FOLDER,exist_ok=True)
os.makedirs(DISTANCE_FOLDER,exist_ok=True)
os.makedirs(DISTANCE_FOLDER_POST,exist_ok=True)


os.makedirs(GESTURE_FOLDER,exist_ok=True)
os.makedirs(GESTURE_FOLDER_POST,exist_ok=True)
os.makedirs(FACES_PATH,exist_ok=True)
os.makedirs(EMOTIONS_FOLDER,exist_ok=True)
os.makedirs(EMOTIONS_FOLDER_FILT,exist_ok=True)
os.makedirs(EMOTIONS_FOLDER_POST,exist_ok=True)

os.makedirs(FEATURES_FOLDER,exist_ok=True)
os.makedirs(FEATURES_POST,exist_ok=True)



#######

video_base_name = os.path.basename(VIDEO_INPUT_PATH).split(".")[0]
VIDEO_RESAMPLED_PATH = os.path.join(RESAMPLED_VIDEOS_FOLDER, video_base_name + ".avi")
video_length = get_video_length(VIDEO_INPUT_PATH)

############### RESAMPLED VIDEO #######################################
print("-"*15 + "RESAMPLING VIDEO" + "-"*15)
#n_frames = resample_video(VIDEO_INPUT_PATH, VIDEO_RESAMPLED_PATH, frame_step=1)
_ = resample_with_ffmpeg(VIDEO_INPUT_PATH, VIDEO_RESAMPLED_PATH, frame_step=25)
n_frames = get_video_length(VIDEO_RESAMPLED_PATH)

print("-"*15 + "DONE" + "-"*15)
########################### TRACKING ########################
script = "src/tracking/segmentation_and_track_main.py"
environment = conda_envs["nvi"]
arguments = {
    "input_video": VIDEO_RESAMPLED_PATH,
    "output_folder": TRACKING_FOLDER
}

# Create the command
command = create_command(script, environment, **arguments)
print("-"*15 + "TRACKING IN PROGRESS" + "-"*15)

# Run the command using bash to ensure environment is activated in the same session
subprocess.run(command, shell=True)
print("-"*15 + "DONE" + "-"*15)

############ SPLIT TEACHERS AND STUDENTS TRACKING ###################
script = "src/tracking/split_teacher_and_students.py"
environment = conda_envs["nvi"]
arguments = {
    "json_path": json_first_frame_labels_path,
    "tracking_folder": os.path.join(TRACKING_FOLDER, video_base_name + "_masks"),
    "teacher_folder_path": os.path.join(TRACKING_FOLDER, video_base_name + "_teacher"),
    "video_id": video_base_name,
}

command = create_command(script, environment, **arguments)
print("-"*15 + "SPLITTING TEACHERS AND STUDENTS" + "-"*15)
subprocess.run(command, shell=True)
print("-"*15 + "DONE" + "-"*15)

##############################################################

#####################   DEPTH       #############################
depth_video_path = os.path.join(RESAMPLED_VIDEOS_FOLDER, video_base_name + ".avi")
script = "src/depth/depth_main.py"
environment = conda_envs["dinov2"]
arguments = {
    "video_path": depth_video_path,
    "output_folder": DEPTH_FOLDER
}

# Create the command
print("-"*15 + "EXTRACTING DEPTH" + "-"*15)

command = create_command(script, environment, **arguments)

# Run the command using bash to ensure environment is activated in the same session
subprocess.run(command, shell=True)
print("-"*15 + "DONE" + "-"*15)

##############################################################

############ EXTRACT DISTANCE ###################
script = "src/classifiers/distance/video_extraction/extract_distance_main.py"
environment = conda_envs["nvi"]
arguments = {
    "depth_folder": os.path.join(DEPTH_FOLDER, video_base_name + ".npy"),
    "segm_folder": os.path.join(TRACKING_FOLDER, video_base_name + "_masks"),
    "teacher_segm": os.path.join(TRACKING_FOLDER, video_base_name + "_teacher"),
    "output_path": os.path.join(DISTANCE_FOLDER, video_base_name + ".csv"),
}

print("-"*15 + " EXTRACTING DISTANCE" + "-"*15)

command = create_command(script, environment, **arguments)
subprocess.run(command, shell=True)
print("-"*15 + "DONE" + "-"*15)

##############################################################

############ EXTRACT GESTURES ###################
script = "src/classifiers/gesture_and_relaxation/model/extract_gestures_main.py"
environment = conda_envs["nvi"]
arguments = {
    "teacher_segm": os.path.join(TRACKING_FOLDER, video_base_name + "_teacher"),
    "output_path": os.path.join(GESTURE_FOLDER, video_base_name + ".csv"),
    "video_path": VIDEO_INPUT_PATH
}

command = create_command(script, environment, **arguments)
subprocess.run(command, shell=True)
print("-"*15 + "DONE" + "-"*15)

##############################################################

############ EXTRACT FACES ###################
script = "src/emotions/extract_faces.py"
environment = conda_envs["hsemotion"]
arguments = {
    "tracking_path": os.path.join(TRACKING_FOLDER, video_base_name + "_teacher"),
    "video_path": VIDEO_RESAMPLED_PATH,
    "faces_path": os.path.join(FACES_PATH, video_base_name)
}

print("-"*15 + " EXTRACTING FACES" + "-"*15)
command = create_command(script, environment, **arguments)
subprocess.run(command, shell=True)
print("-"*15 + "DONE" + "-"*15)
##############################################################
############ EXTRACT EMOTIONS ###################
script = "src/emotions/emotions_main.py"
environment = conda_envs['hsemotion']
arguments = {
    "cropped_face_path": os.path.join(FACES_PATH, video_base_name),
    "output_json": os.path.join(EMOTIONS_FOLDER, video_base_name + ".json"),
    "N_frames": n_frames
}

print("-"*15 + " EXTRACTING FACES" + "-"*15)
command = create_command(script, environment, **arguments)
subprocess.run(command, shell=True)
print("-"*15 + "DONE" + "-"*15)



##################################
#### POSPROCESSING DISTANCE ####

script = "src/classifiers/distance/video_extraction/distance_postprocess_main.py"
environment = conda_envs["nvi"]

arguments = {
    "input_csv": os.path.join(DISTANCE_FOLDER, video_base_name + ".csv"),
    "output_csv": os.path.join(DISTANCE_FOLDER_POST, video_base_name + ".csv"),
    "window_size": 5
}

command = create_command(script, environment, **arguments)

subprocess.run(command, shell=True)

###############################
############ POSTPROCESS GESTURES ###################
script = "src/classifiers/gesture_and_relaxation/model/postprocess_gesture_main.py"
environment = conda_envs["nvi"]
arguments = {
    "input_csv": os.path.join(GESTURE_FOLDER, video_base_name + ".csv"),
    "output_csv": os.path.join(GESTURE_FOLDER_POST, video_base_name + ".csv"),
    "video_length": video_length
}
command = create_command(script, environment, **arguments)
subprocess.run(command, shell=True)
print("-"*15 + " POSTPROCESSING GESTURES" + "-"*15)



###############################
############ FILTER EMOTIONS ###################
script = "src/emotions/filter_emotions_main.py"
environment = conda_envs["nvi"]
arguments = {
    "input_csv": os.path.join(EMOTIONS_FOLDER, video_base_name + ".csv"),
    "output_csv": os.path.join(EMOTIONS_FOLDER_FILT, video_base_name + ".csv"),
    "window_size": 5
}
command = create_command(script, environment, **arguments)

subprocess.run(command, shell=True)
print("-"*15 + " POSTPROCESSING EMOTIONS" + "-"*15)


###############################
############ AVERAGE EMOTIONS ###################
script = "src/emotions/average_emotions_main.py"
environment = conda_envs["nvi"]
arguments = {
    "input_csv": os.path.join(EMOTIONS_FOLDER_FILT, video_base_name + ".csv"),
    "output_csv": os.path.join(EMOTIONS_FOLDER_POST, video_base_name + ".csv")
}

command = create_command(script, environment, **arguments)

subprocess.run(command, shell=True)
print("-"*15 + " POSTPROCESSING EMOTIONS" + "-"*15)


##############################################################
############ MERGE CSVS ###################

script = "src/data_postprocessing/merge_csv_features_main.py"
environment = conda_envs["nvi"]

emotions_csv = os.path.join(EMOTIONS_FOLDER_POST, video_base_name + ".csv")
distance_csv = os.path.join(DISTANCE_FOLDER_POST, video_base_name + ".csv")
gesture_csv = os.path.join(GESTURE_FOLDER_POST, video_base_name + ".csv")
output_csv = os.path.join(FEATURES_POST, video_base_name + ".csv")
arguments = {
    "input": ",".join([emotions_csv, distance_csv, gesture_csv]),
    "output": output_csv
}


print("-"*15 + " MERGING CSVS" + "-"*15)
command = create_command(script, environment, **arguments)
subprocess.run(command, shell=True)

#################### ESTIMATING NVI ################
script = "src/data_postprocessing/make_predictions_main.py"
environment = conda_envs["nvi"]

arguments = {
    "input": os.path.join(FEATURES_POST, video_base_name + ".csv"),
    "model": "checkpoints/mlp_merged_emotions_p493112.joblib",
    "output": NVI_PREDICTIONS_CSV
}

command = create_command(script, environment, **arguments)
subprocess.run(command, shell=True)
