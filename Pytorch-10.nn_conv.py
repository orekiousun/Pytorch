import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 卷积层
"""
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    in_channels表示输入通道，
    out_channels表示输出通道，
    kernel_size表示卷积核大小，
    stride表示每次步数，
    padding表示进行0扩展的位数，
    dilation表示核与核之间的元素距离插值，也叫空洞卷积
"""

dataset = torchvision.datasets.CIFAR10("Dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   #准备数据集
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self,x):
        x = self.conv1(x)   # 将x放入卷积层当中
        return x

tudui = Tudui()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print("output:", output.shape)
    print("input:", imgs.shape)
    # torch.Size([64, 3, 32, 32])
    # print(output.shape)
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) -> [xxx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))   # 为output重新设置大小
    writer.add_images("output", output, step)
    step = step + 1

writer.close()