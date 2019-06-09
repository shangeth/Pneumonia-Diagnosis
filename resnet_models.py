import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
from torch.utils import data
import cv2
from PIL import Image
from collections import OrderedDict

# class ResNet(nn.Module):
#   def __init__(self, mid_fc_dim=200, output_dim=2):
#     super(ResNet, self).__init__()
    
#     self.resnet = models.resnet18(pretrained=True)
#     self.resnet.layer4.requires_grad = True
#     self.features = nn.Sequential(self.resnet.conv1,
#                                   self.resnet.bn1,
#                                   nn.ReLU(),
#                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
#                                   self.resnet.layer1, 
#                                   self.resnet.layer2, 
#                                   self.resnet.layer3, 
#                                   self.resnet.layer4)
    
#     self.avgpool = self.resnet.avgpool
#     self.inp_features = self.resnet.fc.in_features
#     self.fc = nn.Sequential(nn.Linear(self.inp_features, mid_fc_dim),
#                             nn.ReLU(),
#                             nn.Dropout(0.3),
#                             nn.Linear(mid_fc_dim, output_dim),
#                             nn.LogSoftmax(dim=1))
  
#   def forward(self, x):
#     x = self.features(x)
#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     x = self.fc(x)
#     return x
  
def get_densenet_model():
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
        
    inp_features = model.classifier.in_features    
    model.classifier = nn.Sequential(OrderedDict([
        ('fcl1', nn.Linear(inp_features,256)),
        ('dp1', nn.Dropout(0.3)),
        ('r1', nn.ReLU()),
        ('fcl2', nn.Linear(256,32)),
        ('dp2', nn.Dropout(0.3)),
        ('r2', nn.ReLU()),
        ('fcl3', nn.Linear(32,2)),
        ('out', nn.LogSoftmax(dim=1)),
    ]))

    model.load_state_dict(torch.load('model/saved_state.pth', map_location='cpu'))
    return model



#---------------------------------------------------------------------------------------------------------------------------



# ResNet Class
# class ResNet2(nn.Module):
#     def __init__(self, model):
#         super(ResNet2, self).__init__()
        
#         # define the resnet152
#         self.model = model
        
#         # isolate the feature blocks
#         self.features = self.model.features

#         self.avgpool = self.model.avgpool
#         # classifier
#         self.classifier = self.model.fc
        
#         # gradient placeholder
#         self.gradient = None
    
#     # hook for the gradients
#     def activations_hook(self, grad):
#         self.gradient = grad
    
#     def get_gradient(self):
#         return self.gradient
    
#     def get_activations(self, x):
#         return self.features(x)
    
#     def forward(self, x):
#         # extract the features
#         x = self.features(x)
        
#         # register the hook
#         h = x.register_hook(self.activations_hook)
        
#         # complete the forward pass
#         x = self.avgpool(x)
#         x = x.view((x.size(0), -1))
#         x = self.classifier(x)
        
#         return x


class ResNet2(nn.Module):
    def __init__(self, model):
        super(ResNet2, self).__init__()
        
        # define the resnet152
        self.model = model
        
        # isolate the feature blocks
        self.features = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # classifier
        self.classifier = self.model.classifier
        
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