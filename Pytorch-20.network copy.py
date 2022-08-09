import os
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10("Dataset", 
                                       train=True, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   
test_data = torchvision.datasets.CIFAR10("Dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class AN94(nn.Module):
    def __init__(self):
        super(AN94, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


an94 = AN94()
loss_fun = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(an94.parameters(), lr=learning_rate)

epoch = 15
total_train_step = 0   
total_test_step = 0   

# 添加tensorboard，以输出运行结果
writer = SummaryWriter("logs")
test_data_size = len(test_data)

for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i+1))

    # 训练开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = an94(imgs)   
        loss = loss_fun(outputs, targets)   

        # 优化器优化
        optimizer.zero_grad()  
        loss.backward()   #
        optimizer.step()

        # 输出训练结果
        total_train_step = total_train_step + 1 
        if total_train_step % 100 == 0:   # 每100次输出一次结果
            print("训练次数： {}，loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 测试步骤开始
    an94.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  
        for data in test_dataloader:
            imgs, targets = data
            outputs = an94(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()   
            accuracy = (outputs.argmax(1) == targets).sum()   
            total_accuracy = total_accuracy + accuracy

    # 输出测试结果
    print("整体测试上的loss： {}".format(total_test_loss))
    print("整体测试集上的正确率： {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
# tensorboard --logdir=logs





