import torch
import torch.nn as nn
from torchvision import models

class CatDogCNN(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        #load pretrained resnet18
        self.model=models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        
       #replace final layer
        self.model.fc = nn.Linear(
            #numbers of features comming form resnet
            self.model.fc.in_features,
            num_classes
        )
    def forward(self,x):
        x=self.model(x)
        return x