import torch
from torch import nn
import torchvision
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 激活函数
"""
    最终效果有点像给图片打上灰度值
"""

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("Dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   #准备数据集
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()   # inplace表示是否对原来的变量进行直接结果的替换，即把函数执行结果直接赋值给input，默认值为false
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)   # 由于这里inplace是false，所以要将值赋给output
        return output

tudui = Tudui()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output", output, global_step=step)
    step = step + 1

writer.close()
# tensorboard --logdir=logs