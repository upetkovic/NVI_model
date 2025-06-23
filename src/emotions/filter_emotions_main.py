import pandas as pd

def filter_emotions(csv_input_path, csv_output_path, window_size=5):
    df = pd.read_csv(csv_input_path)

    emotion_cols = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", 
                    "Neutral", "Sadness", "Surprise"]

    df_filtered = df.copy()
    half_window = window_size // 2

    for i in range(len(df)):
        if df.loc[i, "emotion"] == "Unknown":
            continue  # Skip Unknowns completely — keep them as-is

        # Define window
        start = max(0, i - half_window)
        end = min(len(df), i + half_window + 1)
        window = df.iloc[start:end]

        # Use only rows with known emotions
        valid_window = window[window["emotion"] != "Unknown"]
        if valid_window.empty:
            continue  # Can't update based on invalid neighbors

        # Smooth: average emotion scores from valid neighbors
        mean_values = valid_window[emotion_cols].mean()
        df_filtered.loc[i, emotion_cols] = mean_values

        # Update emotion label to the dominant one
        new_emotion = mean_values.idxmax()
        df_filtered.loc[i, "emotion"] = new_emotion

    df_filtered.to_csv(csv_output_path, index=False)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Filter emotions in a CSV file using a sliding window.")
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file with emotion scores.")
    parser.add_argument("--output_csv", type=str, help="Path to save the filtered CSV file.")
    parser.add_argument("--window_size", type=int, help="Size of the sliding window for filtering (default: 5).")

    args = parser.parse_args()

    filter_emotions(args.input_csv, args.output_csv, args.window_size)