import numpy as np
import os

folder = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/depth/real_depth1"

# List all .npy files in the folder
npy_files = [f for f in os.listdir(folder) if f.endswith('.npy')]

for file in npy_files:
    file_path = os.path.join(folder, file)
    
    try:
        # Load the .npy file
        data = np.load(file_path)

        # Check if the data is already float16
        if data.dtype != np.float16:
            # Convert the data to float16
            data = data.astype(np.float16)

            # Overwrite the original file
            np.save(file_path, data)

            print(f"Converted and saved: {file}")
        else:
            print(f"File already in float16 format: {file}")

    except Exception as e:
        print(f"Error processing file {file}: {e}")
