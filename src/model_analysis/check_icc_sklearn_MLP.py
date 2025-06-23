import os
import re
import json
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import pearsonr
from datasets.sklearn_data import (
    load_features_from_files,
    read_video_ids,
    load_targets_for_ids,
    normalize_data_with_stats,
    remove_feature_by_name
)
from utils.model_evaluation.helpers import estimate_icc
from utils.training import merge_negative_emotions
# ---- Paths ----
val_ids_path = "/home/uros/Documents/project31/data/NVI/labels/val_ids.txt"
train_ids_path = "/home/uros/Documents/project31/data/NVI/labels/train_ids.txt"

label_csv_path = '/home/uros/Documents/project31/data/NVI/labels/videolabels_merged.csv'
features_folder = "data/outputs/features_postprocessed_2"
normalization_path = "data/config/normalization_parameters_v2.json"
model_dir = "./data/sklearn_saved_models_merged_emotions_normalized_no_real_changes_2"
output_dir = "data/model_analysis/teachers"
os.makedirs(output_dir, exist_ok=True)

# ---- Load features and labels ----

X_val, valid_ids_val, feature_names = load_features_from_files(read_video_ids(val_ids_path), features_folder)
#X_val, valid_ids_val, feature_names = load_features_from_files(samples_paths, features_folder)

y_val, all_ratings, found_ids = load_targets_for_ids(valid_ids_val, label_csv_path)
found_ids_int = [int(x) for x in found_ids]
teacher_IDS = [
    [45,53],
    [99, 107],
    [251, 259],
    [269, 277],
    [278, 286],
    [305, 313],
    [349, 357],
    [358, 366],
    [367, 375]
    ]
teacher_IDS_N = [y-x+ 1 for x, y in teacher_IDS]
# Optional: merge emotion features
negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
X_val, feature_names = merge_negative_emotions(X_val, feature_names, negative_emotions)

# ---- Normalize ----
with open(normalization_path, "r") as f:
    loaded_stats = json.load(f)
X_val = normalize_data_with_stats(X_val, loaded_stats, feature_names)

X_val, feature_names = remove_feature_by_name(X_val, feature_names, 'relative_changes')

# ---- Load Raters ----
df = pd.read_csv(label_csv_path)
rater0 = all_ratings[:, 0]
rater1 = all_ratings[:, 1]
rater2 = all_ratings[:, 2]

def make_rater_array(*raters):
    return np.array(raters).T

def parse_hidden_layers_from_filename(filename):
    match = re.search(r"mlp_layers_([0-9_]+)_r_", filename)
    if not match:
        raise ValueError("Filename format is incorrect")
    layer_str = match.group(1)
    return [int(x) for x in layer_str.split("_")]

# ---- Load and Evaluate Models ----
model_paths = [
    os.path.join(model_dir, fname)
    for fname in os.listdir(model_dir)
    if fname.endswith(".joblib") and fname.startswith("mlp_layers_")
]
model_paths.sort()
model_paths = ["data/sklearn_saved_models_merged_emotions_normalized_no_real_changes/mlp_layers_270_100_10_r_0p493112.joblib"]
#print(model_paths)
for path in model_paths:
    model = load(path)
    preds = model.predict(X_val)
    
    corr, p = pearsonr(y_val, preds)
    if corr > 0.43:
        preds = preds * 10000  # optional rescaling

        rater_array = make_rater_array(rater0, rater1, rater2, preds)
        icc_values = estimate_icc(rater_array)
        if icc_values[-1] < 0.6:
            continue  # skip weak ICCs

        print(path)
        print(f"ICC: r0, r1, r2, out: {icc_values}")
        print(f"Model: {os.path.basename(path)}, Pearson r: {corr:.4f}, p-value: {p:.4f}")
        hidden_layers = parse_hidden_layers_from_filename(os.path.basename(path))
        print(f"Model with hidden layers {hidden_layers}")

        for combo, label in [
            ((rater0, rater1, rater2), "r0, r1, r2"),
            ((rater0, rater1, preds), "r0, r1, out"),
            ((rater0, preds, rater2), "r0, out, r2"),
            ((preds, rater1, rater2), "out, r1, r2"),
            ((rater0, rater1, rater2, preds), "r0, r1, r2, out"),
        ]:
            icc_values = estimate_icc(make_rater_array(*combo))
            print(f"ICC: {label}: {icc_values}")

        cnt = 0
        for i, teacher_n in enumerate(teacher_IDS_N):
            corr, p = pearsonr(y_val[cnt:cnt+teacher_n], preds[cnt:cnt+teacher_n])
            mae = np.mean(np.abs(y_val[cnt:cnt+teacher_n]*10000 - preds[cnt:cnt+teacher_n]))
            medianae = np.median(np.abs(y_val[cnt:cnt+teacher_n]*10000 - preds[cnt:cnt+teacher_n]))

            #print(f"Teacher {i+1} ({teacher_n} samples): Pearson r: {mae:.4f}, p-value: {p:.4f}")
            print(f"Teacher {i+1} ({teacher_n} samples): Pearson r: {mae:.4f}, {medianae:.4f}, {corr:.4f}, p-value: {p:.4f}")

            cnt += teacher_n

        #print(preds, y_val
        y_val = y_val*10000
        print(f"MAE: {np.mean(np.abs(preds - y_val)):.4}")
        print(f"std: {np.std(np.abs(preds - y_val)):.4}")

     
        print("\n" + "="*50 + "\n")

                ######################### SAB
        #y_val_rater_extended = np.hstack((y_val_rater.copy(), y_pred_val.reshape(-1, 1)))
        new_ground_truth = np.median(rater_array, axis=1).reshape(-1, 1)
        new_ground_truth = new_ground_truth.ravel()

        correlation_coefficient, p_value = pearsonr(y_val, preds)
        print(f"Pearson correlation coefficient - MODEL vs target: {correlation_coefficient}")
        print(f"P-value: {p_value}")

        correlation_coefficient, p_value = pearsonr(new_ground_truth, rater_array[:, 0])
        print(f"Pearson correlation coefficient - rater0 vs NEW target: {correlation_coefficient}")
        print(f"P-value: {p_value}")

        correlation_coefficient, p_value = pearsonr(new_ground_truth, rater_array[:, 1])
        print(f"Pearson correlation coefficient - rater1 vs NEW target: {correlation_coefficient}")
        print(f"P-value: {p_value}")

        correlation_coefficient, p_value = pearsonr(new_ground_truth, rater_array[:, 2])
        print(f"Pearson correlation coefficient - rater2 vs NEW target: {correlation_coefficient}")
        print(f"P-value: {p_value}")

        correlation_coefficient, p_value = pearsonr(new_ground_truth, rater_array[:, 3])
        print(f"Pearson correlation coefficient - MODEL vs NEW target: {correlation_coefficient}")
        print(f"P-value: {p_value}")
        ###########################


import matplotlib.pyplot as plt
import numpy as np

y_val = y_val/10000
# Collect true and predicted values for each teacher
true_values = []
pred_values = []
cnt = 0
for teacher_n in teacher_IDS_N:
    true_values.append((y_val[cnt:cnt+teacher_n] * 1).tolist())
    pred_values.append((preds[cnt:cnt+teacher_n]/10000).tolist())
    cnt += teacher_n

import matplotlib.pyplot as plt
import numpy as np

# Compute per-sample signed errors for each teacher
errors = [np.array(p) - np.array(t) for t, p in zip(true_values, pred_values)]

plt.figure(figsize=(5.5, 4))
plt.boxplot(errors, labels=[f"Teacher {i+1}" for i in range(len(errors))],
            showmeans=True, patch_artist=True)
plt.axhline(0, color='black', linestyle='--')  # zero-error line
plt.ylabel("Prediction Error (Predicted - True)")
plt.title("Signed Prediction Errors per Teacher")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_whiskers.png"), dpi=300, bbox_inches='tight')



# Compute per-sample signed errors for each teacher
errors = [abs(np.array(p) - np.array(t)) for t, p in zip(true_values, pred_values)]

plt.figure(figsize=(5.5, 4))
plt.boxplot(errors, labels=[f"Teacher {i+1}" for i in range(len(errors))],
            showmeans=True, patch_artist=True)
plt.axhline(0, color='black', linestyle='--')  # zero-error line
plt.ylabel("Absolute Error")
plt.title("Absolute Error per Teacher")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "MAE_whiskers.png"), dpi=300, bbox_inches='tight')
