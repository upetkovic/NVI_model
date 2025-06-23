import os
import json
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
from datasets.sklearn_data import (
    normalize_data_with_stats,
    load_features_from_files,
    read_video_ids,
    load_targets_for_ids,
    remove_feature_by_name
)
from utils.training import merge_negative_emotions


def adjust_emotions_to_sum_one(
    emotion_values: np.ndarray,
    target_idx: int
) -> np.ndarray:
    """
    Adjust emotion values so that their sum stays 1 after changing one target emotion.
    """
    adjusted = emotion_values.copy()
    target_value = adjusted[target_idx]
    rest_sum = emotion_values.sum() - target_value

    remaining = 1.0 - target_value

    if rest_sum == 0:
        n_rest = len(adjusted) - 1
        if n_rest > 0:
            for i in range(len(adjusted)):
                if i != target_idx:
                    adjusted[i] = 0
                else:
                    adjusted[target_idx] = 1.0
    else:
        scale = remaining / rest_sum
        for i in range(len(adjusted)):
            if i == target_idx:
                adjusted[i] = target_value
            else:
                adjusted[i] *= scale

    return adjusted


# === Configuration paths ===
val_ids_path = "labels/val_ids.txt"
train_ids_path = "labels/train_ids.txt"

features_folder = "data/outputs/features_postprocessed_2"
features_folder_additional = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NVI_additional/features_postprocessed_2"

features_stats_path = "data/config/normalization_parameters_v2.json"

normalize_path = "data/config/normalization_parameters_v2.json"
model_path = "./data/sklearn_saved_models_merged_emotions_normalized_no_real_changes/mlp_layers_270_100_10_r_0p493112.joblib"

output_dir = "data/model_analysis/feature_effects"
os.makedirs(output_dir, exist_ok=True)


# Create a list of strings from '00430' to '00760'
id_list = [f"{i:05d}" for i in range(403, 761)]
# === Load data ===
val_paths = read_video_ids(val_ids_path) #+ read_video_ids(train_ids_path)
X_val, valid_ids_val, feature_names = load_features_from_files(val_paths, features_folder)
#X_val, valid_ids_val, feature_names = load_features_from_files(id_list, features_folder_additional)

negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
X_val, feature_names = merge_negative_emotions(X_val, feature_names, negative_emotions)

X_val, feature_names = remove_feature_by_name(X_val, feature_names, 'relative_changes')

# === Load model and stats ===
model = load(model_path)

with open(features_stats_path, "r") as f:
    features_stats = json.load(f)

with open(normalize_path, "r") as f:
    normalize_stats = json.load(f)

print(features_stats)

# === Normalize the validation set once ===
X_val_norm = normalize_data_with_stats(X_val.copy(), normalize_stats, feature_names)

# === Emotion setup ===
emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
emotions = ['neg_emotion_sum', 'Happiness', 'Neutral']

feature_names_dic = {name:name for name in feature_names}
feature_names_dic['neg_emotion_sum'] = 'Neg. emotions'
feature_names_dic['relative_time'] = 'Gestures'
feature_names_dic['mean_distance'] = 'Distance avg.'
feature_names_dic['std_distance'] = 'Distance SD'
feature_names_dic['face_visibility'] = 'Face vis.'




emotion_indices = [feature_names.index(em) for em in emotions]


# === Run feature analysis ===
N = 111
plt.figure()

for idx, feature in enumerate(feature_names):
    if feature not in features_stats:
        print(f"Skipping {feature} (no stats found)")
        continue

    mean = features_stats[feature]['mean']
    std = features_stats[feature]['std']
    dx = np.linspace(-2*std, 2*std, N)

    preds_all = []
    X_mod = X_val.copy()
    X_tmp = normalize_data_with_stats(X_mod, normalize_stats, feature_names)
    preds_orig = model.predict(X_tmp)
    if feature in emotions:
        target_idx = emotions.index(feature)

        for i in range(N):

            val = np.clip(mean + dx[i], 0, 1)
            X_mod = X_val.copy()
            X_mod[:, idx] += dx[i]
            X_mod[:, idx] = np.clip(X_mod[:, idx], 0, 1)
            for j in range(X_mod.shape[0]):


                original_emotions = X_mod[j, emotion_indices]
                adjusted_emotions = adjust_emotions_to_sum_one(original_emotions, target_idx)
                X_mod[j, emotion_indices] = adjusted_emotions

                if adjusted_emotions.sum()<0.9999 or adjusted_emotions.sum()>1.0001:
                    print(adjusted_emotions.sum())
            X_tmp = normalize_data_with_stats(X_mod, normalize_stats, feature_names)

            preds = model.predict(X_tmp)
            preds_all.append(preds)



    else:
        for i in range(N):
            X_mod = X_val.copy()
            X_mod[:, idx] += dx[i]
            X_mod[:, idx] = np.clip(X_mod[:, idx], 0, 1)
            X_tmp = normalize_data_with_stats(X_mod, normalize_stats, feature_names)
            preds = model.predict(X_tmp)
            #preds_all.append(preds - preds_orig)
            preds_all.append(preds)

    preds_all = np.array(preds_all)  # shape: (N, num_samples)
    preds_mean = preds_all.mean(axis=1)


    plt.figure()

    plt.plot(dx / std, preds_mean, label=feature)
    plt.xlabel(f"{feature_names_dic[feature]} (delta from mean)")
    plt.ylabel("Average model prediction")
    plt.title(f"Effect of varying {feature_names_dic[feature]}")
    plt.savefig(os.path.join(output_dir, f"feature_effect_{feature_names_dic[feature]}.png"))
    plt.close()


plt.figure()

for idx, feature in enumerate(feature_names):
    if feature not in features_stats:
        print(f"Skipping {feature} (no stats found)")
        continue

    mean = features_stats[feature]['mean']
    std = features_stats[feature]['std']
    dx = np.linspace(-2*std, 2*std, N) / 1

    preds_all = []
    X_mod = X_val.copy()
    X_tmp = normalize_data_with_stats(X_mod, normalize_stats, feature_names)
    preds_orig = model.predict(X_tmp)
    if feature in emotions:
        target_idx = emotions.index(feature)

        for i in range(N):

            #val = np.clip(mean + dx[i], 0, 1)
            X_mod = X_val.copy()
            X_mod[:, idx] += dx[i]
            X_mod[:, idx] = np.clip(X_mod[:, idx], 0, 1)
            for j in range(X_mod.shape[0]):
                original_emotions = X_mod[j, emotion_indices]
                adjusted_emotions = adjust_emotions_to_sum_one(original_emotions, target_idx)
                X_mod[j, emotion_indices] = adjusted_emotions

                if adjusted_emotions.sum()<0.9999 or adjusted_emotions.sum()>1.0001:
                    print(adjusted_emotions.sum())
            X_tmp = normalize_data_with_stats(X_mod, normalize_stats, feature_names)

            preds = model.predict(X_tmp)
            preds_all.append(preds)



    else:
        for i in range(N):
            X_mod = X_val.copy()
            X_mod[:, idx] += dx[i]
            X_mod[:, idx] = np.clip(X_mod[:, idx], 0, 1)
            X_tmp = normalize_data_with_stats(X_mod, normalize_stats, feature_names)
            preds = model.predict(X_tmp)
            preds_all.append(preds)

    preds_all = np.array(preds_all)  # shape: (N, num_samples)
    preds_mean = preds_all.mean(axis=1)

    # Print prediction at dx = 0 for debug
    print(f"{feature}: dx=0 prediction = {preds_mean[N // 2]}")

    plt.plot(dx / std, preds_mean - preds_mean[N // 2], label=feature_names_dic[feature])

plt.xlabel("Change from original value (in SD units)")
plt.ylabel("Average model prediction change")
plt.title("Effect of varying features on prediction")
plt.legend(ncol=2)
plt.grid(True)

# Save the combined plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_effect_combined_500.pdf"), dpi=500, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "feature_effect_combined_500.png"), dpi=500, bbox_inches='tight')

#plt.savefig("feature_effect_combined.pdf", dpi=330, bbox_inches='tight')

plt.close()
