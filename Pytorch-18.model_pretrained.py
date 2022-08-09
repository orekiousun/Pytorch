import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)   # vgg16是事先写好的训练模型

print('vgg16_true:', vgg16_true) 
print('vgg16_false', vgg16_false)

vgg16_true.add_module("add_linear", nn.Linear(1000, 10))   # 为最后加上一个线性层
print('vgg16_true:', vgg16_true)

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))   # 在classifier中加线性层
print('vgg16_true:', vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)   # 修改某一层
print('vgg16_false', vgg16_false)