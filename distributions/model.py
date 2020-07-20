"""
Convolutional model to extract features
Fully connected layers to do regression
Dropout to prevent over fitting the model
"""
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    """
    initalializing the model.
    Kernel_size: 3x3
    Filter_size: [32, 64, 128, 256]
    Hidden_size: [64, 512, 1024]
    Final_outputsize = 10
    """
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(36864, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(.1)
        self.drop2 = nn.Dropout(.2)

    def forward(self, x):
        """
        Args:
            x: Inputs preprocessed images.
        Returns: Prediction of all 5 points
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
