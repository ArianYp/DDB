
import torch
import torch.nn as nn
import torchvision

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)
    
    def forward (self, X):
        X = self.model(X)
        return X