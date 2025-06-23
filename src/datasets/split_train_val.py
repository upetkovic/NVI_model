import csv

csv_path = '/home/uros/Documents/project31/data/NVI/labels/videolabels_merged.csv'
train_output_path = '/home/uros/Documents/project31/data/NVI/labels/train_ids.txt'
val_output_path = '/home/uros/Documents/project31/data/NVI/labels/val_ids.txt'

train_ids = []
val_ids = []

with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    immediacy_fields = [field for field in reader.fieldnames if field.startswith("Immediacy")]

    for row in reader:
        # Skip the row if any Immediacy field has value "SKIP"
        if any(row[col].strip().upper() == "SKIP" for col in immediacy_fields):
            continue

        video_id = row['videoID']
        if row['train_val'] == 'train':
            train_ids.append(video_id)
        elif row['train_val'] == 'val':
            val_ids.append(video_id)

# Sort and write to files
train_ids.sort()
val_ids.sort()

with open(train_output_path, 'w') as f:
    for vid in train_ids:
        f.write(f"{vid}\n")

with open(val_output_path, 'w') as f:
    for vid in val_ids:
        f.write(f"{vid}\n")

print("Filtered and saved train/val IDs.")
print(f"Total train IDs: {len(train_ids)}")
print(f"Total val IDs: {len(val_ids)}")
