import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from dataloader import DistanceDataset
from model import FinetuneResnet, SimpleCNN, FinetuneResnetNew
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import itertools
import os


#NEW
lr_s = [0.001, 0.005, 0.0002, 0.0001]
batch_size_s = [32, 16, 8, 48]
num_epochs_s = [160]
size_in= [[800, 600, 150], [800, 600, 100], [800, 600, 200], [600, 300, 120]]


parameters = itertools.product(lr_s)

parameters = list(parameters) * 7

batch_size = 128
batch_size = 90

lr = 0.001

ratings_path_val = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_VAL.csv"
ratings_path_train = "/home/uros/Documents/project31/data/classifiers_frames/human_ratings/labels_and_features_TRAIN.csv"

depth_path = "/home/uros/Documents/project31/data/classifiers_frames/depth_frames/real_depth"
segm_path = "/home/uros/Documents/project31/data/classifiers_frames/segm_frames_humans/real_segm"

weights_dir = f"/media/uros/b497d77f-967a-48ec-aea0-5c352966318c/new_data/project31/data/NI_classifiers/distance/weights"

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

dataset_val = DistanceDataset(depth_path=depth_path, segm_path=segm_path, ratings_path=ratings_path_val, std_th=500000)
dataset_train = DistanceDataset(depth_path=depth_path, segm_path=segm_path, ratings_path=ratings_path_train, std_th=1600)
mean_output = dataset_train.mean_output

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Your device: {device}')

size_in = [512, 512, 64]
model = FinetuneResnet()
#model = FinetuneResnetNew(mlp_size=size_in)
#model = SimpleCNN()
model = model.float()
model = model.to(device)

criterion = nn.MSELoss()






# Fixed inputs
fixed_image_inputs, _ = next(iter(val_loader))
fixed_image_inputs = fixed_image_inputs.to(device)


# Training loop
num_epochs = 80


for hyper_param in parameters:

    # TensorBoard setup
    writer = SummaryWriter(log_dir="./runs2")

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

        model.train()
        train_losses = [] # Create a list to store individual batch losses
        mean_losses_train = []

        # Wrap your train_loader with tqdm for progress display
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to device
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, targets)
            
            # Compute mean prediction loss during training
            mean_preds = torch.full_like(targets, mean_output)
            mean_loss = criterion(mean_preds, targets)
            #mean_loss = F.l1_loss(mean_preds, targets)
            mean_losses_train.append(mean_loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())  # Append the batch loss
            # Log training loss to TensorBoard
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)

        avg_train_loss = sum(train_losses) / len(train_losses)  # Compute average training loss for the epoch
        avg_mean_train_loss = sum(mean_losses_train) / len(mean_losses_train)

        # Validation loop
        model.eval()
        val_losses = []
        mean_losses_val = []
        targets_list = []
        outputs_list = []

        # Wrap your val_loader with tqdm for progress display
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validating"):
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to device
            with torch.no_grad():
                outputs = model(inputs).view(-1)  
                loss = criterion(outputs, targets)
                #loss = F.l1_loss(outputs, targets)
                val_losses.append(loss.item())

                # Compute mean prediction loss during validation
                mean_preds_val = torch.full_like(targets, mean_output)
                mean_val_loss = criterion(mean_preds_val, targets)
                #mean_val_loss = F.l1_loss(mean_preds_val, targets)

                mean_losses_val.append(mean_val_loss.item())

                [targets_list.append(x.item()) for x in targets]
                [outputs_list.append(x.item()) for x in outputs]
            
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_mean_vall_loss = sum(mean_losses_val) / len(mean_losses_val)
        # Log validation loss to TensorBoard
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)

        # Print the average training and validation losses for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Correlation: {np.corrcoef(targets_list, outputs_list)[0, 1]}")


        if best_loss_val > avg_val_loss:
            best_loss_val = avg_val_loss

        if avg_val_loss < 0.0122:
            save_id = f"loss{int(100000*avg_val_loss):06}corr{int(1000*np.corrcoef(targets_list, outputs_list)[0, 1]):04}"      
            weights_path = os.path.join(weights_dir, save_id + ".pth")
            torch.save(model.state_dict(), weights_path)
            dic_path = os.path.join(weights_dir, save_id + ".json")


        if epoch == 0:
            print(f"Mean Prediction Loss (Train): {avg_mean_train_loss:.4f}, Mean Prediction Loss (Validation): {avg_mean_vall_loss:.4f}")

        # Log predictions to TensorBoard every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                fixed_outputs = model(fixed_image_inputs)
                # Here you log the mean or histogram of outputs, as per your needs.
                writer.add_histogram('Predictions', fixed_outputs, epoch)
        
    writer.add_hparams(
            {"lr": lr, "bsize": batch_size, "Num epochs":num_epochs,"model_size0": size_in[0],
                "model_size1": size_in[1], "model_size2": size_in[2]},
            {     
                "loss_best": best_loss_val,
            },
        )

    writer.close()
