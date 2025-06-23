import pandas as pd

def average_emotions(csv_input_path, csv_output_path):

    df = pd.read_csv(csv_input_path)

    # Define emotion columns
    emotion_cols = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", 
                    "Neutral", "Sadness", "Surprise"]

    output_dic = {}
    unknown_rows = len(df[df["emotion"] == "Unknown"])
    known_rows = len(df) - unknown_rows
    for emotion in emotion_cols:
        output_dic[emotion] = len(df[df["emotion"] == emotion]) / known_rows
    output_dic["face_visibility"] = known_rows/len(df)
    # Write output_dic to a CSV file
    output_df = pd.DataFrame([output_dic], columns=emotion_cols + ["face_visibility"])
    output_df.to_csv(csv_output_path, index=False)
        
    
if __name__ == "__main__":
    #average_emotions(csv_input_path="data/outputs/emotions_filtered/00000.csv", csv_output_path="00000_bin.csv")
    import argparse

    parser = argparse.ArgumentParser(description="Average emotions in a CSV file.")
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file with emotion scores.")
    parser.add_argument("--output_csv", type=str, help="Path to save the averaged emotions CSV file.")

    args = parser.parse_args()

    average_emotions(args.input_csv, args.output_csv)