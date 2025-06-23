import pandas as pd
from joblib import load
import numpy as np
import os
import csv
import argparse
from datasets.sklearn_data import normalize_data_with_stats, remove_feature_by_name
import json
from utils.model_evaluation.helpers import parse_hidden_layers_from_filename
import torch
from utils.training import merge_negative_emotions

def estimate_NVI(input_path, model_path, output_path):
    basename = os.path.basename(input_path).split(".")[0]
    
    # Load and preprocess input data
    df = pd.read_csv(input_path)
    feature_names = list(df.columns)
    X = np.array([df.iloc[0].values])

    feature_names = list(df.columns)
    negative_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness']
    X, feature_names = merge_negative_emotions(X, feature_names, negative_emotions)

    with open("data/config/normalization_parameters_v2.json", "r") as f:
        loaded_stats = json.load(f)
    X = normalize_data_with_stats(X, loaded_stats, feature_names)

    X, feature_names = remove_feature_by_name(X, feature_names, 'relative_changes')

    # Load scaler and model
    model = load(model_path)
    nvi = model.predict(X)[0]


    # Write to output file
    file_exists = os.path.exists(output_path)
    with open(output_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["basename", "NVI"])
        writer.writerow([basename, nvi])


def main():
    
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input binarized feature CSV")
    parser.add_argument("--model", required=True, help="Path to trained model .joblib file")
    parser.add_argument("--output", required=True, help="CSV file to append prediction results to")

    args = parser.parse_args()
    estimate_NVI(args.input, args.model, args.output)

if __name__ == "__main__":
    main()
