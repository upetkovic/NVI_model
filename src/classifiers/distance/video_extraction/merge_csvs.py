import pandas as pd
import os

FROM_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/gesture/results"
FROM_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/distance/results_part2"
FROM_PATH = "/home/uros/Documents/project31/predictions_part2enet_b0_8_best_vgaf"
FROM_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/gesture/results_part2"

TO_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_dist_emot"
TO_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/gesture/results_part2"
TO_PATH = "/home/uros/Documents/project31/predictions_part2enet_b0_8_best_vgaf"
TO_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_emot_dist_part2"

NEW_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_dist_emot_gesture3"
NEW_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_emot_dist_part2"
NEW_PATH = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/features_all_emot_dist_gesture_part2"

if not os.path.exists(NEW_PATH):
    os.makedirs(NEW_PATH)

csv_files = [f for f in os.listdir(FROM_PATH) if f.endswith('.csv')]

print(csv_files)

for csv_id in csv_files:
    from_path = os.path.join(FROM_PATH, csv_id)
    to_path = os.path.join(TO_PATH, csv_id)
    write_path = os.path.join(NEW_PATH, csv_id)

    if not os.path.exists(from_path):
        continue

    if not os.path.exists(to_path):
        continue  
    # Load the two CSV files
    csv1 = pd.read_csv(to_path)
    csv2 = pd.read_csv(from_path)
    csv1 = csv1[:750]
    csv2 = csv2[:750]

    #
    # Ensure the two CSVs have the same number of rows or that you handle differing row counts
    if len(csv1) != len(csv2):
        raise ValueError("CSV files have different number of rows. Please ensure they match or handle it accordingly.")

    # Copy a column from csv2 to csv1
    # Suppose you want to copy the column named "ColumnNameToCopy" from csv2 to csv1
    #csv1['gesture_cont'] = csv2['gesture_cont']
    csv1['gesture_cont'] = csv2['gesture_cont']

    # Save the merged CSV
    csv1.to_csv(write_path, index=False)