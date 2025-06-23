import os
import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr
from joblib import dump
from datasets.sklearn_data import (
    load_features_from_files,
    read_video_ids,
    load_targets_for_ids,
    normalize_data_with_stats,
    remove_feature_by_name,
    merge_negative_emotions
)



# ----- Paths -----
val_ids_path = "labels/val_ids.txt"
train_ids_path = "labels/train_ids.txt"
label_csv_path = 'labels/videolabels_merged.csv'
features_folder = "data/outputs/features_postprocessed_2"
norm_stats_path = "data/config/normalization_parameters_v2.json"
save_dir = "./data/sklearn_saved_models_merged_emotions_normalized_no_real_changes"

os.makedirs(save_dir, exist_ok=True)

# ----- Load data -----
X_val, valid_ids_val, feature_names = load_features_from_files(read_video_ids(val_ids_path), features_folder)
X_train, valid_ids_train, _ = load_features_from_files(read_video_ids(train_ids_path), features_folder)
y_val, _, _ = load_targets_for_ids(valid_ids_val, label_csv_path)
y_train, _, _ = load_targets_for_ids(valid_ids_train, label_csv_path)

# Optional: merge emotion features
negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']

X_train, _ = merge_negative_emotions(X_train, feature_names, negative_emotions)
X_val, feature_names = merge_negative_emotions(X_val, feature_names, negative_emotions)

# ----- Normalize -----
with open(norm_stats_path, "r") as f:
    loaded_stats = json.load(f)
X_train = normalize_data_with_stats(X_train, loaded_stats, feature_names)
X_val = normalize_data_with_stats(X_val, loaded_stats, feature_names)

X_train, feature_names_new = remove_feature_by_name(X_train, feature_names, 'relative_changes')
X_val, feature_names = remove_feature_by_name(X_val, feature_names, 'relative_changes')


# ----- Configurations -----

hidden_layer_configs = [
    [300, 100, 10]
]


# ----- Training loop -----
for layers in hidden_layer_configs:
    #print(f"\nTraining MLPRegressor with layers {layers}")
    model = MLPRegressor(
        hidden_layer_sizes=tuple(layers),
        max_iter=5000,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        tol=1e-5,
        n_iter_no_change=20,
    )

    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    # Pearson correlation
    r_val, _ = pearsonr(y_val, y_pred_val)
    print(f"Pearson r (val): {r_val:.4f}")

    # Save 
    layer_str = "_".join(str(x) for x in layers)
    corr_str = f"{r_val:.6f}".replace(".", "p")
    model_path = os.path.join(save_dir, f"mlp_layers_{layer_str}_r_{corr_str}.joblib")
    dump(model, model_path)
    print(f"✅ Saved model to: {model_path}")

