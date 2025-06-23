import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from dataset import GestureDataset
from model import FinetuneResnet, SimpleCNN, FinetuneResnetNew
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
import csv
import glob
from scipy.stats import pearsonr

#NEW
lr_s = [0.001, 0.005, 0.0002, 0.0001]
batch_size_s = [32, 16, 8, 48]
num_epochs_s = [160]
size_in= [[800, 600, 150], [800, 600, 100], [800, 600, 200], [600, 300, 120]]


def plot_correlation(list1, list2, filename='correlation_plot.png'):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")

    # Calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(list1, list2)[0, 1]

    # Plot the data
    plt.scatter(list1, list2, label=f'Correlation = {correlation_coefficient:.2f}')
    plt.title('Scatter plot with Correlation')
    plt.xlabel('List 1')
    plt.ylabel('List 2')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(filename, format='png')
    plt.close()



parameters = itertools.product(lr_s)

parameters = list(parameters) * 23

batch_size = 82
lr = 0.001

ratings_path_val = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL.csv"
ratings_path_train = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_TRAIN.csv"

depth_path = "/home/uros/Documents/project31/data/classifiers_frames/depth_frames/real_depth"
segm_path = "/home/uros/Documents/project31/data/classifiers_frames/segm_frames_humans/real_segm"
frame_path = "/home/uros/Documents/project31/data/classifiers_frames/Talis_frames15_v2"

weights_dir = f"/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI_classifiers/gesture/weights"

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

dataset_val = GestureDataset(depth_path=depth_path, segm_path=segm_path, frame_path=frame_path, ratings_path=ratings_path_val, std_th=500000)

val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Your device: {device}')

size_in = [512, 512, 64]
model = FinetuneResnet()
#model = FinetuneResnetNew(mlp_size=size_in)
#model = SimpleCNN()
model = model.float()
model = model.to(device)

loss_dir = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI_classifiers/gesture/weights"
weights_path =  "loss002807corr0838acc849.pth"
weights_path = os.path.join(loss_dir, weights_path)

segm_dir = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI/semantic_segmentation/humans/real_segm7"
videos_dir = "/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/TalisTeacherVideos"
videos_path = glob.glob(os.path.join(videos_dir, "*.avi"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FinetuneResnet()
model = model.float()
model.load_state_dict(torch.load(weights_path))
model.to(device)


criterion = nn.MSELoss()






# Fixed inputs
fixed_image_inputs, _, _ = next(iter(val_loader))
fixed_image_inputs = fixed_image_inputs.to(device)


# Training loop
num_epochs = 1

def get_binaray_acc(outputs, targets, th=0.1):
    output_np = np.array(outputs)
    target_np = np.array(targets)

    output_true = output_np > th
    target_true = target_np > th

    return np.sum(output_true == target_true) / len(output_true)


for hyper_param in [parameters[0]]:

    # TensorBoard setup

    lr = hyper_param[0]
    lr = 0.001
    best_loss_val = 1000
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        '''
        if epoch == 0:
            optimizer.param_groups[0]['lr'] = 0.002  # Initial learning rate
        elif epoch == 5:
            optimizer.param_groups[0]['lr'] = 0.001  # Reduce learning rate after 20 epochs
        elif epoch == 50:
            optimizer.param_groups[0]['lr'] = 0.0006  # Reduce learning rate further after 60 epochs
        elif epoch == 120:
            optimizer.param_groups[0]['lr'] = 0.0002  # Reduce learning rate further after 60 epochs
        '''

    

        # Validation loop
        model.eval()
        val_losses = []
        mean_losses_val = []
        targets_list = []
        outputs_list = []
        imgIDS_list = []

        # Wrap your val_loader with tqdm for progress display
        for inputs, targets, imgIDS in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validating"):
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to device
            with torch.no_grad():
                outputs = model(inputs).view(-1)  
                loss = criterion(outputs, targets)
                #loss = F.l1_loss(outputs, targets)
                val_losses.append(loss.item())

                # Compute mean prediction loss during validation
                #mean_val_loss = F.l1_loss(mean_preds_val, targets)


                [targets_list.append(x.item()) for x in targets]
                [outputs_list.append(x.item()) for x in outputs]
                [imgIDS_list.append(x) for x in imgIDS]

        # Print the average training and validation losses for the current epoch
        #print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Correlation: {np.corrcoef(targets_list, outputs_list)[0, 1]}")
        correlation_coefficient, p_value = pearsonr(targets_list, outputs_list)

        print(f"Pearson correlation coefficient: {correlation_coefficient}")
        print(f"P-value: {p_value}")
        with open('gesture_validation.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header (optional)
            writer.writerow(["id", "model", "target"])

            # Write the lists to the CSV file
            for item1, item2, item3 in zip(imgIDS_list, outputs_list, targets_list):
                writer.writerow([item1, item2, item3])


