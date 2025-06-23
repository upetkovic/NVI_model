import argparse
import pandas as pd

def concat_csv_columns(file_paths, output_path):
    # Read all CSVs and store them in a list
    dfs = [pd.read_csv(path) for path in file_paths]

    # Concatenate columns (axis=1)
    merged_df = pd.concat(dfs, axis=1)

    # Save to output
    merged_df.to_csv(output_path, index=False)
    print(f"Concatenated CSV saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Comma-separated list of CSV files to concatenate")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    file_paths = [f.strip() for f in args.input.split(",")]
    concat_csv_columns(file_paths, args.output)

if __name__ == "__main__":
    main()
