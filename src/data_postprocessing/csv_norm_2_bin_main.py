import os
import pandas as pd
import argparse

emotion_cols = [
    'Anger', 'Contempt', 'Disgust', 'Fear',
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]

def norm_2_bin(input_path, output_path):
    df = pd.read_csv(input_path)

    # Get the column name with the maximum value for each row, only considering the emotion columns
    max_columns = df[emotion_cols].idxmax(axis=1)

    # Create a dataframe of zeros with the shape of the emotion columns
    binary_df = pd.DataFrame(0, index=df.index, columns=emotion_cols)

    # Set the maximum column to 1 for each row
    for row, col in enumerate(max_columns):
        binary_df.at[row, col] = 1

    # Update the original dataframe with the binary values
    df[emotion_cols] = binary_df

    df['gesture_cont'] = (df['gesture_cont'] > 0.5).astype(int)
    df['distance'] = (df['distance'] > 0.5).astype(int)


    # Save result
    df.to_csv(output_path, index=False)
    print(f"Binarized CSV saved to: {output_path}")

def main():
    #norm_2_bin("/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_dist_emot_gesture3/tmp_norm/00281.csv", "test_bin.csv")
    #norm_2_bin("data/outputs/features_norm/00281.csv", "data/outputs/features_bin/00281.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input normalized CSV")
    parser.add_argument("--output", required=True, help="Path to output binarized CSV")
    args = parser.parse_args()

    norm_2_bin(args.input, args.output)

if __name__ == "__main__":
    main()
