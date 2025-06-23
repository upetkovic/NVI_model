import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from joblib import dump
from datasets.sklearn_data import (
    load_features_from_files,
    read_video_ids,
    load_targets_for_ids,
    normalize_data_with_stats,
    remove_feature_by_name
)

# Optional: helper to merge emotions (left unchanged)
def merge_negative_emotions(X, feature_names, negative_emotions, new_feature_name='neg_emotion_sum'):
    neg_indices = [feature_names.index(em) for em in negative_emotions]
    neg_sum = X[:, neg_indices].sum(axis=1, keepdims=True)
    keep_indices = [i for i in range(X.shape[1]) if i not in neg_indices]
    X_clean = X[:, keep_indices]
    X_new = np.hstack([X_clean, neg_sum])
    feature_names_new = [feature_names[i] for i in keep_indices] + [new_feature_name]
    return X_new, feature_names_new

# ----- Paths -----
val_ids_path = "labels/val_ids.txt"
train_ids_path = "labels/train_ids.txt"
label_csv_path = 'labels/videolabels_merged.csv'
features_folder = "data/outputs/features_postprocessed_2"
norm_stats_path = "data/config/normalization_parameters_v2.json"
save_dir = "./data/sklearn_saved_models_emotions_linear"
os.makedirs(save_dir, exist_ok=True)

# ----- Load data -----
X_val, valid_ids_val, feature_names = load_features_from_files(read_video_ids(val_ids_path), features_folder)
X_train, valid_ids_train, _ = load_features_from_files(read_video_ids(train_ids_path), features_folder)
y_val, _, _ = load_targets_for_ids(valid_ids_val, label_csv_path)
y_train, _, _ = load_targets_for_ids(valid_ids_train, label_csv_path)

# Optional: merge emotion features
negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
#X_train, _ = merge_negative_emotions(X_train, feature_names, negative_emotions)
#X_val, feature_names = merge_negative_emotions(X_val, feature_names, negative_emotions)

# ----- Normalize -----
with open(norm_stats_path, "r") as f:
    loaded_stats = json.load(f)
X_train = normalize_data_with_stats(X_train, loaded_stats, feature_names)
X_val = normalize_data_with_stats(X_val, loaded_stats, feature_names)

X_train, _ = remove_feature_by_name(X_train, feature_names, 'relative_changes')
X_val, feature_names = remove_feature_by_name(X_val, feature_names, 'relative_changes')

# ----- Train Linear Regression -----
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)

# ----- Evaluation -----
r_val, _ = pearsonr(y_val, y_pred_val)
print(f"📊 Pearson r (val): {r_val:.4f}")

# ----- Save the model -----
model_path = os.path.join(save_dir, f"linear_model_r_{r_val:.6f}".replace(".", "p") + ".joblib")
dump(model, model_path)
print(f"✅ Saved linear model to: {model_path}")
