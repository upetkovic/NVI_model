import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets.sklearn_data import (
    load_features_from_files,
    read_video_ids,
    remove_feature_by_name
)
from joblib import load
import json
from utils.training import merge_negative_emotions
import os
import math

# === Paths ===
val_ids_path = "labels/train_ids.txt"
model_path = "data/sklearn_saved_models_emotions_normalized_no_real_changes/mlp_layers_300_100_12_r_0p478942.joblib"
features_folder = "data/outputs/features_postprocessed_2"
features_stats_path = "data/config/normalization_parameters_v2.json"
normalize_path = "data/config/normalization_parameters_v2.json"
output_dir = "data/model_analysis/features_histograms"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
val_paths = read_video_ids(val_ids_path)
X_val, valid_ids_val, feature_names = load_features_from_files(val_paths, features_folder)
X_val, feature_names = remove_feature_by_name(X_val, feature_names, 'relative_changes')

negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
X_val_2, feature_names_2 = merge_negative_emotions(X_val.copy(), feature_names, negative_emotions)
# Add 'neg_emotion_sum' to X_val
neg_emotion_sum = X_val_2['neg_emotion_sum'].values if isinstance(X_val_2, pd.DataFrame) else X_val_2[:, feature_names_2.index('neg_emotion_sum')]
X_val = np.column_stack([X_val, neg_emotion_sum])
feature_names.append('neg_emotion_sum')

print(X_val.shape)
# === Load model and stats ===
model = load(model_path)
with open(features_stats_path, "r") as f:
    features_stats = json.load(f)
with open(normalize_path, "r") as f:
    normalize_stats = json.load(f)

# === Feature name mapping (clean labels for plots) ===
feature_names_dic = {name: name for name in feature_names}
feature_names_dic['relative_time'] = 'Gestures'
feature_names_dic['mean_distance'] = 'Distance avg.'
feature_names_dic['std_distance'] = 'Distance SD'
feature_names_dic['face_visibility'] = 'Face vis.'
feature_names_dic['neg_emotion_sum'] = 'Neg. Emotions'

# === Prepare DataFrame ===
X_val_df = pd.DataFrame(X_val, columns=[feature_names_dic.get(name, name) for name in feature_names])

# === Plotting settings ===
num_features = X_val_df.shape[1]
cols = 4
rows = math.ceil(num_features / cols)

fig_width_inch = 7.2   # full width for 2-column journal
fig_height_inch = rows * 2.2  # row height scales with content

# Font sizes for journal (10–11 pt)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

fig, axes = plt.subplots(
    rows, cols,
    figsize=(fig_width_inch, fig_height_inch),
    gridspec_kw={
        'wspace': 0.5,   # horizontal space between columns
        'hspace': 0.5    # vertical space between rows
    }
)
axes = axes.flatten()



emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
n_bins = 18

for i, col in enumerate(X_val_df.columns):
    data = X_val_df[col].dropna().values
    weights = np.ones_like(data) / len(data)
    bin_edges = np.linspace(0, 1, n_bins)

    # Plot histogram
    axes[i].hist(data, bins=bin_edges, weights=weights, edgecolor='black', alpha=0.75)
    axes[i].set_title(col, pad=4)
    axes[i].xaxis.set_label_coords(0.5, -0.18)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Rel. Frequency")
    # Add y-label only to the first subplot in each row
    if i % cols != 0:
        axes[i].set_ylabel("")
    
    if i < 8 or i>11:
        axes[i].set_ylim(0, 1)
    axes[i].tick_params(axis='x', rotation=30)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Layout adjustments
plt.tight_layout(h_pad=1.4, w_pad=1.1)
fig.subplots_adjust(top=0.93)
plt.suptitle('Baseline and Refined Feature Distributions', fontsize=11)

# === Save figure ===
fig.savefig(os.path.join(output_dir, "normalized_feature_histograms.png"), bbox_inches='tight', dpi=300)
# Optional: vector export for journals
fig.savefig(os.path.join(output_dir, "normalized_feature_histograms.pdf"), bbox_inches='tight', pad_inches=0.001)
