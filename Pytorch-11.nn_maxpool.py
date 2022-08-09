import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 最大池化层
"""
    最大池化指在核内选择最大的数作为输出
    ceil_mode为True时表示可能核内数字不足，但仍进行最大化保留
    可以保留数据特征的同时减少数据量
    最终结果就类似照片像素降低，被打码
"""

dataset = torchvision.datasets.CIFAR10("Dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   #准备数据集
dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                     [0, 1, 2, 3, 1],
                     [1, 2, 1, 0, 0],
                     [5, 2, 3, 1, 1],
                     [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)
        # self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui() 
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1
    
writer.close()