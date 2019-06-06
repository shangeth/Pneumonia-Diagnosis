import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
from torch.utils import data
import cv2
from PIL import Image


class ResNet(nn.Module):
  def __init__(self, mid_fc_dim=200, output_dim=2):
    super(ResNet, self).__init__()
    
    self.resnet = models.resnet18(pretrained=True)
    self.resnet.layer4.requires_grad = True
    self.features = nn.Sequential(self.resnet.conv1,
                                  self.resnet.bn1,
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                  self.resnet.layer1, 
                                  self.resnet.layer2, 
                                  self.resnet.layer3, 
                                  self.resnet.layer4)
    
    self.avgpool = self.resnet.avgpool
    self.inp_features = self.resnet.fc.in_features
    self.fc = nn.Sequential(nn.Linear(self.inp_features, mid_fc_dim),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(mid_fc_dim, output_dim),
                            nn.LogSoftmax(dim=1))
  
  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
  



#---------------------------------------------------------------------------------------------------------------------------



# ResNet Class
class ResNet2(nn.Module):
    def __init__(self, model):
        super(ResNet2, self).__init__()
        
        # define the resnet152
        self.model = model
        
        # isolate the feature blocks
        self.features = self.model.features

        self.avgpool = self.model.avgpool
        # classifier
        self.classifier = self.model.fc
        
        # gradient placeholder
        self.gradient = None
    
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forward(self, x):
        # extract the features
        x = self.features(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        
        return x


