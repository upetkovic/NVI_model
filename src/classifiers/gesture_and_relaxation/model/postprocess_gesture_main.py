import pandas as pd
def postprocess_gesture(input_csv_path, output_csv_path, video_length):

    # Read the input CSV file
    df = pd.read_csv(input_csv_path)
    num_rows = len(df)
    gesture_time = len(df[df["gesture_cont"] > 0.1])

    df["gesture_bin"] = (df["gesture_cont"] > 0.1)*1.0
    # Calculate the number of changes from 1 to 0 in the "gesture_bin" timeseries
    changes = ((df["gesture_bin"].shift(1) == 1) & (df["gesture_bin"] == 0)).sum()

    # If the last value is 1, count it as an additional change
    if df["gesture_bin"].iloc[-1] == 1:
        changes += 1

    relative_time = gesture_time / num_rows if num_rows > 0 else 0

    # Write the relative_time to the output CSV file
    output_df = pd.DataFrame({"relative_time": [relative_time], "relative_changes": [changes/video_length]})
    output_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    #postprocess_gesture("data/outputs/gesture/00034.csv", "00034_gestures.csv", video_length=30)
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess gesture data from a CSV file.")
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file with gesture data.")
    parser.add_argument("--output_csv", type=str, help="Path to save the postprocessed gesture data.")
    parser.add_argument("--video_length", type=int, help="Length of the video in seconds (default: 30).")

    args = parser.parse_args()

    postprocess_gesture(args.input_csv, args.output_csv, args.video_length)
