import os
import json
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
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

# Define a scorer for Pearson correlation
def pearson_scorer(y_true, y_pred):
    if y_true.ndim == 1:
        return pearsonr(y_true, y_pred)[0]
    else:
        return np.mean([pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])

pearson_cv = make_scorer(pearson_scorer)

# ----- Paths -----
val_ids_path = "labels/val_ids.txt"
train_ids_path = "labels/train_ids.txt"
label_csv_path = 'labels/videolabels_merged.csv'
features_folder = "data/outputs/features_postprocessed_2"
norm_stats_path = "data/config/normalization_parameters_v2.json"
save_dir = "./data/sklearn_saved_models_emotions_svr_best"
os.makedirs(save_dir, exist_ok=True)

# ----- Load and normalize data -----
X_val, valid_ids_val, feature_names = load_features_from_files(read_video_ids(val_ids_path), features_folder)
X_train, valid_ids_train, _ = load_features_from_files(read_video_ids(train_ids_path), features_folder)
y_val, _, _ = load_targets_for_ids(valid_ids_val, label_csv_path)
y_train, _, _ = load_targets_for_ids(valid_ids_train, label_csv_path)

with open(norm_stats_path, "r") as f:
    loaded_stats = json.load(f)


# Optional: merge emotion features
negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
X_train, _ = merge_negative_emotions(X_train, feature_names, negative_emotions)
X_val, feature_names = merge_negative_emotions(X_val, feature_names, negative_emotions)


X_train = normalize_data_with_stats(X_train, loaded_stats, feature_names)
X_val = normalize_data_with_stats(X_val, loaded_stats, feature_names)

X_train, _ = remove_feature_by_name(X_train, feature_names, 'relative_changes')
X_val, feature_names = remove_feature_by_name(X_val, feature_names, 'relative_changes')

# ----- Grid Search with RBF Kernel -----
base_svr = SVR(kernel='rbf')
model = MultiOutputRegressor(base_svr) if y_train.ndim == 2 else base_svr

param_grid = {
    'estimator__C': [0.1, 1, 10],
    'estimator__gamma': ['scale', 0.01, 0.001],
    'estimator__epsilon': [0.01, 0.1, 0.2]
} if y_train.ndim == 2 else {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'epsilon': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(model, param_grid, cv=3, scoring=pearson_cv, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred_val = best_model.predict(X_val)

# ----- Evaluation -----
if y_val.ndim == 1:
    r_val, _ = pearsonr(y_val, y_pred_val)
else:
    r_val = np.mean([pearsonr(y_val[:, i], y_pred_val[:, i])[0] for i in range(y_val.shape[1])])

print(f"📊 Best Pearson r (val): {r_val:.4f}")
print(f"Best parameters: {grid.best_params_}")

# ----- Save the model -----
corr_str = f"{r_val:.6f}".replace(".", "p")
model_path = os.path.join(save_dir, f"best_svr_r_{corr_str}.joblib")
dump(best_model, model_path)
print(f"✅ Saved best SVR model to: {model_path}")
