import torch.nn as nn
import torchvision.models as models


class ApneaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ApneaClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.model(x)
