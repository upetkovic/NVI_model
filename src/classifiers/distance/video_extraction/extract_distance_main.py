import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from classifiers.distance.model.model import FinetuneResnet
import csv
from PIL import Image

def preprocess_image(image, scale_factor = 0.33):
    H = int(image.shape[1] * scale_factor)
    W = int(image.shape[2] * scale_factor)
    image = image.unsqueeze(0)
    image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)
    image = image.squeeze(0)
    return image

def extract_distance(depth_path, segm_folder_path,segm_folder_teacher_path, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/distance_loss001154corr0552.pth"
    model = FinetuneResnet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.float()

    #depth_path = os.path.join(depth_folder, file_id)
    #segm_folder_path = os.path.join(segm_folder, file_id.split(".")[0] + "_masks")
    #segm_folder_teacher_path = os.path.join(segm_folder, file_id.split(".")[0] + "_teacher")

    #csv_id = file_id.split(".")[0] + ".csv"
    depth_images = np.load(depth_path)

    if not os.path.exists(segm_folder_teacher_path):
        pass

    mask_list = [f for f in os.listdir(segm_folder_teacher_path) if os.path.isfile(os.path.join(segm_folder_teacher_path, f))]
    mask_list.sort()
    masks_all = []
    [masks_all.append(int(mask.split(".")[0])) for mask in mask_list]

    #loop over frames
    batch_size = len(masks_all)
    batch_size = 108
    batch_size = int(np.ceil(750 / 7))

    image_batch = torch.zeros(size=(batch_size, 3, 237, 422 ))
    num_frames = max(masks_all) + 1
    results_batches = []

    # Loop over each batch
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        current_batch_size = batch_end - batch_start

        image_batch = torch.zeros(size=(current_batch_size, 3, 237, 422))
        frame_ids_list = []

        # Process each frame in the current batch
        for batch_idx, frame_id in enumerate(range(batch_start, batch_end)):
            frame_id = int(frame_id)
            frame_ids_list.append(frame_id)
            depth_image = depth_images[frame_id]

            # Process segmentation teacher image
            segm_teacher_path = os.path.join(segm_folder_teacher_path, f"{frame_id:05d}" + ".png")
            with Image.open(segm_teacher_path) as img:
                grayscale_img = img.convert('L')
                teacher_mask = np.array(grayscale_img)
                teacher_mask = (teacher_mask > 0).astype(int)

            # Process segmentation student image
            segm_path = os.path.join(segm_folder_path, f"{frame_id:05d}" + ".png")
            with Image.open(segm_path) as img:
                grayscale_img = img.convert('L')
                student_mask = np.array(grayscale_img)
                student_mask = (student_mask > 0).astype(int)
                student_mask = student_mask - teacher_mask

            # Prepare input image
            input_image = np.zeros(shape=(3, depth_image.shape[0], depth_image.shape[1]))
            input_image[2, :, :] = depth_image / 6 
            input_image[0, :, :] = teacher_mask
            input_image[1, :, :] = student_mask
            image = torch.from_numpy(input_image)
            image = preprocess_image(image.float())
            image_batch[batch_idx, :, :, :] = image


            """
            img = image.detach().cpu().numpy()  # shape: (3, W, H)
            img = np.transpose(img, (1, 2, 0))
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

            Image.fromarray(img).save("output_path"+str(frame_id) + ".png")
            """

        # Apply model to the current batch
        model.eval()
        with torch.no_grad():        
            results = model(image_batch.to(device)).to('cpu').numpy()
            [results_batches.append(result) for result in results]

    frames_out = list(range(num_frames))
    results_out = np.array(results_batches).squeeze().tolist()

    #csv_path = os.path.join(output_folder, csv_id)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "distance"])  # Column headers
        writer.writerows(zip(list(frames_out), list(results_out)))
        

if __name__ == "__main__":
    """
    extract_distance(
        depth_folder="./data/outputs/depth",
        segm_folder="./data/outputs/tracking",
        output_folder='data/outputs/distance',
        file_id="00281" + ".npy"
    )
    """
    """
    extract_distance(
        depth_path="/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/depth/real_depth7/00281.npy",
        segm_folder_path="outputs/00281_colored_masks",
        segm_folder_teacher_path="outputs/00281_colored_teacher",
        csv_path="data/outputs/distance/00281_test.csv"
    )
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_folder", required=True, help="Path npy depth file.")
    parser.add_argument("--segm_folder", required=True, help="Path to all masks.")
    parser.add_argument("--teacher_segm", required=True, help="Path to teacher masks.")
    parser.add_argument("--output_path", required=True, help="Path to the output csv")
    args = parser.parse_args()

    extract_distance(
        depth_path=args.depth_folder,
        segm_folder_path=args.segm_folder,
        segm_folder_teacher_path=args.teacher_segm,
        csv_path=args.output_path
    )
