import pandas as pd
import os
import numpy as np
from utils.model_evaluation.helpers import estimate_icc

def make_rater_array(*raters):
    return np.array(raters).T

validation_file_path = "data/classifiers/distance_validation.csv"
#validation_file_path = "data/classifiers/gesture_validation.csv"

ratings_path_val = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL.csv"


validation_df = pd.read_csv(validation_file_path)
validation_dict = validation_df.set_index("id").to_dict(orient="index")

labels_df = pd.read_csv(ratings_path_val)
labels_dict = labels_df.set_index("imgID").to_dict(orient="index")

keys_features = ["distance0", "distance1", "distance2"]
#keys_features = [ "gesture0", "gesture1", "gesture2"]
for key in validation_dict:
    for feature in keys_features:
        validation_dict[key][feature] = labels_dict[key][feature]/10000


updated_validation_df = pd.DataFrame.from_dict(validation_dict, orient="index").reset_index().rename(columns={"index": "id"})

print("DISTANCE ICCs")

all_ratings = np.array(updated_validation_df[keys_features + ["model"]])
rater0 = all_ratings[:, 0]
rater1 = all_ratings[:, 1]
rater2 = all_ratings[:, 2]
preds = all_ratings[:, 3]
for combo, label in [
    ((rater0, rater1, rater2), "r0, r1, r2"),
    ((rater0, rater1, preds), "r0, r1, out"),
    ((rater0, preds, rater2), "r0, out, r2"),
    ((preds, rater1, rater2), "out, r1, r2"),
    ((rater0, rater1, rater2, preds), "r0, r1, r2, out"),
]:
    icc_values = estimate_icc(make_rater_array(*combo))
    print(f"ICC: {label}: {icc_values}")



validation_file_path = "data/classifiers/gesture_validation.csv"
ratings_path_val = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL.csv"


validation_df = pd.read_csv(validation_file_path)
validation_dict = validation_df.set_index("id").to_dict(orient="index")

labels_df = pd.read_csv(ratings_path_val)
labels_dict = labels_df.set_index("imgID").to_dict(orient="index")

keys_features = [ "gesture0", "gesture1", "gesture2"]
for key in validation_dict:
    for feature in keys_features:
        validation_dict[key][feature] = labels_dict[key][feature]/10000


updated_validation_df = pd.DataFrame.from_dict(validation_dict, orient="index").reset_index().rename(columns={"index": "id"})


print("GESTURE ICCs")


all_ratings = np.array(updated_validation_df[keys_features + ["model"]])
rater0 = all_ratings[:, 0]
rater1 = all_ratings[:, 1]
rater2 = all_ratings[:, 2]
preds = all_ratings[:, 3]
for combo, label in [
    ((rater0, rater1, rater2), "r0, r1, r2"),
    ((rater0, rater1, preds), "r0, r1, out"),
    ((rater0, preds, rater2), "r0, out, r2"),
    ((preds, rater1, rater2), "out, r1, r2"),
    ((rater0, rater1, rater2, preds), "r0, r1, r2, out"),
]:
    icc_values = estimate_icc(make_rater_array(*combo))
    print(f"ICC: {label}: {icc_values}")