import os
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from joblib import dump
from datasets.sklearn_data import (
    load_features_from_files,
    read_video_ids,
    load_targets_for_ids,
    normalize_data_with_stats,
    remove_feature_by_name
)
from utils.training import merge_negative_emotions

# ----- Pearson scorer -----
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
save_dir = "./data/sklearn_saved_models_emotions_rf_gridsearch_2"
os.makedirs(save_dir, exist_ok=True)

# ----- Load and preprocess data -----
X_val, valid_ids_val, feature_names = load_features_from_files(read_video_ids(val_ids_path), features_folder)
X_train, valid_ids_train, _ = load_features_from_files(read_video_ids(train_ids_path), features_folder)
y_val, _, _ = load_targets_for_ids(valid_ids_val, label_csv_path)
y_train, _, _ = load_targets_for_ids(valid_ids_train, label_csv_path)

with open(norm_stats_path, "r") as f:
    loaded_stats = json.load(f)
X_train = normalize_data_with_stats(X_train, loaded_stats, feature_names)
X_val = normalize_data_with_stats(X_val, loaded_stats, feature_names)
negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
X_val, _ = merge_negative_emotions(X_val, feature_names, negative_emotions)
X_train, feature_names = merge_negative_emotions(X_train, feature_names, negative_emotions)


X_train, _ = remove_feature_by_name(X_train, feature_names, 'relative_changes')
X_val, feature_names = remove_feature_by_name(X_val, feature_names, 'relative_changes')

# ----- Random Forest with GridSearchCV -----
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 4]
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=pearson_cv,
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred_val = best_model.predict(X_val)

# ----- Evaluation -----
if y_val.ndim == 1:
    r_val, _ = pearsonr(y_val, y_pred_val)
else:
    r_val = np.mean([pearsonr(y_val[:, i], y_pred_val[:, i])[0] for i in range(y_val.shape[1])])

print(f"📊 Best Pearson r (val): {r_val:.4f}")
print(f"🧪 Best parameters: {grid.best_params_}")

# ----- Save model -----
corr_str = f"{r_val:.6f}".replace(".", "p")
model_path = os.path.join(save_dir, f"rf_grid_model_r_{corr_str}.joblib")
dump(best_model, model_path)
print(f"✅ Saved best Random Forest model to: {model_path}")
