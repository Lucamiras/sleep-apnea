import torch
import torch.nn as nn
import torchaudio
import torchvision


class ApneaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ApneaClassifier, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
