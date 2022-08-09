import torch
from torch import nn
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader

# 线性层
"""
    最终效果有点像给图片打上灰度值
"""

dataset = torchvision.datasets.CIFAR10("Dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   #准备数据集
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(in_features=196608, out_features=10)
        
    def forward(self, input):
        output = self.linear1(input)   # 由于这里inplace是false，所以要将值赋给output
        return output

tudui = Tudui()

for data in dataloader:
    imgs, target = data
    print(imgs.shape)

    # output = torch.reshape(imgs, (1,1,1,-1))
    output = torch.flatten(imgs)   # 将张量展平的两种方法
    print(output.shape)

    output = tudui(output)
    print(output.shape)
