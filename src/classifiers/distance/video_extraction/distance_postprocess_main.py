import pandas as pd

def distance_postprocess(input_csv_path, output_csv_path, window_size):

    # Read the input CSV file
    df = pd.read_csv(input_csv_path)

    # Calculate the mean of the "distance" column
    mean_distance = df["distance"].mean()
    # Apply a rolling window filter with a window size of 3 and calculate the mean
    df["filtered_distance"] = df["distance"].rolling(window=window_size, center=True, min_periods=1).median()
    std_distance = df["filtered_distance"].std()

    # Write the relative_time to the output CSV file
    output_df = pd.DataFrame({"mean_distance": [mean_distance], "std_distance": [std_distance]})
    output_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    #distance_postprocess("data/outputs/distance/00034.csv", "00034_distance.csv", window_size=5)
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess distance data from a CSV file.")
    parser.add_argument("--input_csv", type=str, required=True, help="...")
    parser.add_argument("--output_csv", type=str, required=True, help="...")
    parser.add_argument("--window_size", type=int, default=3, help="...")


    args = parser.parse_args()

    distance_postprocess(args.input_csv, args.output_csv, args.window_size)