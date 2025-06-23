import os
import pandas as pd
import numpy as np
import json



def remove_feature_by_name(X, feature_names, feature_to_remove):
    """
    Removes a feature from the feature matrix and feature names by its name.

    Args:
        X (np.ndarray): Feature matrix (shape: [n_samples, n_features])
        feature_names (list of str): Names of all features in X
        feature_to_remove (str): Name of the feature to remove

    Returns:
        X_new (np.ndarray): Feature matrix with the feature removed
        feature_names_new (list of str): Updated feature names
    """
    if feature_to_remove not in feature_names:
        raise ValueError(f"Feature '{feature_to_remove}' not found in feature_names.")
    idx = feature_names.index(feature_to_remove)
    X_new = np.delete(X, idx, axis=1)
    feature_names_new = feature_names[:idx] + feature_names[idx+1:]
    return X_new, feature_names_new


def load_features_from_files(video_ids, feature_folder):
    """
    Loads one-row-per-video feature files into a matrix for scikit-learn models.

    Args:
        video_ids (list): List of video IDs as strings (e.g. ["00001", "00002"])
        feature_folder (str): Path to the folder with .csv files (each = one video)

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        valid_ids (list): List of video IDs successfully loaded
        feature_names (list): Column names (only from the first successfully loaded file)
    """
    X = []
    valid_ids = []
    feature_names = None

    for vid in video_ids:
        file_path = os.path.join(feature_folder, f"{vid}.csv")
        if not os.path.isfile(file_path):
            print(f"Missing file: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            if df.shape[0] != 1:
                raise ValueError(f"File {vid}.csv has {df.shape[0]} rows, expected 1.")
            X.append(df.iloc[0].values)
            valid_ids.append(vid)
            if feature_names is None:
                feature_names = list(df.columns)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return np.array(X), valid_ids, feature_names

def read_video_ids(txt_file):
    """Reads a .txt file with video filenames and strips extensions."""
    with open(txt_file, 'r') as f:
        ids = [line.strip().replace('.avi', '') for line in f if line.strip()]
    return ids



def load_targets_for_ids(video_ids, label_csv_path):
    """
    Loads target values (median and individual ratings) for a list of video IDs.

    Args:
        video_ids (list of str): List of video IDs like '00045', '00046', ...
        label_csv_path (str): Path to 'videolabels_merged.csv'

    Returns:
        y (np.ndarray): Median target values (shape: n_samples,)
        all_ratings (np.ndarray): Raw rater values (shape: n_samples, n_raters)
        found_ids (list): Video IDs actually found and used (same order as y)
    """
    df = pd.read_csv(label_csv_path)

    # Extract base ID (without .avi)
    df['videoID_base'] = df['videoID'].str.replace('.avi', '', regex=False)

    # Get all Immediacy columns
    immediacy_cols = [col for col in df.columns if col.startswith("Immediacy")]

    # Convert all Immediacy values to numeric (SKIP -> NaN)
    df[immediacy_cols] = df[immediacy_cols].apply(pd.to_numeric, errors='coerce')

    # Compute median (ignoring NaN)
    df['target'] = df[immediacy_cols].median(axis=1, skipna=True)
    #df['target'] = df[immediacy_cols].mean(axis=1, skipna=True)


    # Filter to given video_ids
    id_set = set(video_ids)
    df_filtered = df[df['videoID_base'].isin(id_set)]

    # Ensure correct order
    df_filtered = df_filtered.set_index('videoID_base').loc[video_ids]

    # Final outputs
    y = df_filtered['target'].values
    all_ratings = df_filtered[immediacy_cols].values
    found_ids = df_filtered.index.tolist()

    return y/10000, all_ratings, found_ids


def merge_negative_emotions(X, feature_names, negative_emotions, new_feature_name='neg_emotion_sum'):
    """
    Sums specified negative emotion columns and returns a new feature matrix with those columns removed
    and the new summed column appended at the end.

    Args:
        X (np.ndarray): Input feature matrix (shape: [n_samples, n_features])
        feature_names (list of str): Names of all features in X
        negative_emotions (list of str): Names of features to be summed
        new_feature_name (str): Name of the new summed feature

    Returns:
        X_new (np.ndarray): Modified feature matrix
        feature_names_new (list of str): Updated list of feature names
    """
    # Get indices of the negative emotion columns
    neg_indices = [feature_names.index(em) for em in negative_emotions]

    # Sum the selected emotion columns
    neg_sum = X[:, neg_indices].sum(axis=1, keepdims=True)

    # Keep only the non-negative emotion columns
    keep_indices = [i for i in range(X.shape[1]) if i not in neg_indices]
    X_clean = X[:, keep_indices]

    # Append the new feature
    X_new = np.hstack([X_clean, neg_sum])

    # Update feature names
    feature_names_new = [feature_names[i] for i in keep_indices] + [new_feature_name]

    return X_new, feature_names_new


def normalize_data(X, all_features_names, normalized_features_names):
    stats = {}
    for feature in all_features_names:
        if feature in normalized_features_names:
            idx = all_features_names.index(feature)
            mean = np.mean(X[:, idx])
            std = np.std(X[:, idx])
            stats[feature] = {'mean': mean, 'std': std}
        else:
            stats[feature] = {'mean': 0, 'std': 1}

    X_normalized = X.copy()
    for feature in normalized_features_names:
        if feature in all_features_names:
            idx = all_features_names.index(feature)
            X_normalized[:, idx] = (X[:, idx] - stats[feature]['mean']) / (stats[feature]['std'] if stats[feature]['std'] != 0 else 1)
    return X_normalized, stats

def normalize_data_with_stats(X, stats, all_features_names):
    """
    Normalize the data X using precomputed statistics.

    Args:
        X (np.ndarray): Input feature matrix (shape: [n_samples, n_features])
        stats (dict): Dictionary with mean and std for each feature
        all_features_names (list of str): Names of all features in X

    Returns:
        X_normalized (np.ndarray): Normalized feature matrix
    """
    X_normalized = X.copy()
    for feature, stat in stats.items():
        if feature in all_features_names:
            idx = all_features_names.index(feature)
            mean = stat['mean']
            std = stat['std']
            X_normalized[:, idx] = (X[:, idx] - mean) / (std if std != 0 else 1)
    return X_normalized

if __name__ == "__main__":
    video_ids = read_video_ids("/home/uros/Documents/project31/data/NVI/labels/train_ids.txt")
    feature_folder = "data/outputs/features_postprocessed_2"

    X, valid_ids, all_feature_names = load_features_from_files(video_ids, feature_folder)
    feature_names = ['face_visibility', 'mean_distance', 'std_distance', 'relative_time', 'relative_changes']
    negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
    X, feature_names = merge_negative_emotions(X, all_feature_names, negative_emotions)
    X_normalized, stats = normalize_data(X, feature_names, feature_names)
    # Save stats to JSON
    stats_path = "data/config/normalization_parameters_v2_merged_emotions.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Example: Load stats from JSON
    # Load stats as a Python dictionary
    with open(stats_path, "r") as f:
        loaded_stats = json.load(f)
    print("Loaded normalization stats:")
    print(loaded_stats['Anger'])
    X_normalized = normalize_data_with_stats(X, loaded_stats, all_feature_names)
    print("Normalized data shape:", X_normalized.shape)
    y, all_ratings, found_ids = load_targets_for_ids(valid_ids, '/home/uros/Documents/project31/data/NVI/labels/videolabels_merged.csv' )
    #print(y, found_ids, all_ratings)
    a = 3