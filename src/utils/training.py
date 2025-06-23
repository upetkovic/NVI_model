import os
import numpy as np
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