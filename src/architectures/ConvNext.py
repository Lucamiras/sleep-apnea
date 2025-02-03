import torch
import torch.nn as nn
import torchvision.models as models


class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNeXt, self).__init__()
        self.convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights)

        for param in self.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

