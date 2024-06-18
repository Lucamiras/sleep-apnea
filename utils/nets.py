import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = f.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 16 * 16)  # Flatten
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        return x
