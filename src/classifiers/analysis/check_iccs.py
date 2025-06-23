import pandas as pd
import os
import numpy as np
from utils.model_evaluation.helpers import estimate_icc

#from utils.model_evaluation.helpers import estimate_icc


def make_rater_array(*raters):
    return np.array(raters).T
ratings_path_val = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL_and_TRAIN.csv"
df = pd.read_csv(ratings_path_val)
keys_features = ["distance0", "distance1", "distance2"]

all_ratings = np.array(df[keys_features + ["distance2"]])
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
