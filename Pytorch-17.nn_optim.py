import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("Dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   # 加载数据集
dataloader = DataLoader(dataset, batch_size=64)   # 加载数据集

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )   # 也可以通过Sequential的方式将执行的一系列层的功能封装起来

    def forward(self, x):
        x = self.model1(x)   # 也可以利用Sequential一起调用
        return x

tudui = Tudui()
loss = nn.CrossEntropyLoss()   # 定义交叉商

# step1 定义优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)   # parameter表示参数，lr表示学习速率

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        # step2 梯度清零
        optim.zero_grad()  
        # step3 反向传播
        result_loss.backward()   
        # step4 对参数进行调整优化
        optim.step()   
        running_loss = running_loss + result_loss
        print(result_loss)


# tensorboard --logdir=logs