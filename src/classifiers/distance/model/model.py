import torch
from torchvision import models, transforms
from torch import nn
import torch.nn.functional as F


class FinetuneResnet(nn.Module):
    def __init__(self, num_resnet_features=512, input_size=[512, 3000, 100],device='cpu'):
        super(FinetuneResnet, self).__init__()

        self.model = models.resnet18(pretrained=True).to(device)
        self.fc_res1 = nn.Linear(512, 512)  # Updated to match ResNet18's output size
        self.fc_res2 = nn.Linear(512, num_resnet_features)
        self.dropout = nn.Dropout(0.3)

        # model
        self.fc1 = nn.Linear(num_resnet_features, 3000)
        
        self.fc2 = nn.Linear(3000, 100)

        # Change this to output 1 value for regression
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc_res1(x))
        x = self.fc_res2(x)

        f = self.model.relu(self.fc1(x))
        f = self.model.relu(self.fc2(f))

        # Return raw output for regression, without applying softmax or sigmoid
        return self.fc3(f)

class FinetuneResnetNew(nn.Module):
    def __init__(self, mlp_size=[512, 512, 64], device='cpu'):
        super(FinetuneResnetNew, self).__init__()

        self.model = models.resnet18(pretrained=True).to(device)
        self.fc_res1 = nn.Linear(512, mlp_size[0])  # Updated to match ResNet18's output size
        self.fc_res2 = nn.Linear(mlp_size[0], mlp_size[1])
        self.dropout = nn.Dropout(0.3)

        # model
        self.fc1 = nn.Linear(mlp_size[1], mlp_size[2])
        
        self.fc2 = nn.Linear(mlp_size[2], 1)



    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.relu(self.fc_res1(x))
        x = self.model.relu(self.fc_res2(x))

        f = self.model.relu(self.fc1(x))
        #f = self.model.relu(self.fc2(f))

        # Return raw output for regression, without applying softmax or sigmoid
        return self.fc2(f)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 29 * 52, 512)  # Adjusted the dimensions
        self.fc2 = nn.Linear(512, 1) # Regression output
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(-1, 64 * 29 * 52)  # Adjusted the dimensions
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation for regression output
        return x


if __name__ == "__main__":
    num_resnet_features = 2048
    model = FinetuneResnet(num_resnet_features=num_resnet_features)

    r_input = torch.rand(size=(32, 3, 360, 640))
    model(r_input)