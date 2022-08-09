import torchvision 
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10("Dataset", 
                                       train=True, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   # 加载训练数据集
test_data = torchvision.datasets.CIFAR10("Dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)   # 加载测试数据集

# 打印长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
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


# 若采用第二种方法搭建神经网络, 可以直接引用写好的模型
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()   # 利用gpu训练


# 损失函数
loss_fun = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fun = loss_fun.cuda()   # 利用gpu训练

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0   # 记录训练的次数
total_test_step = 0   # 记录测试次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")
for i in range(epoch):
    start_time = time.time()
    print("--------第 {} 轮训练开始--------".format(i+1))

    # 训练步骤开始
    # tudui.train()   # 如果有一些特定的层要使用train，如dropout
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()   # 利用gpu训练
        outputs = tudui(imgs)   # 经过网络模型
        loss = loss_fun(outputs, targets)   # 将预测的输出和真实的targets做对比

        # 优化器优化模型
        optimizer.zero_grad()   # 梯度清零
        loss.backward()   # 反向传播
        optimizer.step()

        # 输出训练结果
        total_train_step = total_train_step + 1 
        if total_train_step % 500 == 0:   # 每500次输出一次结果
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数： {}，loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():   # with使网络模型中没有梯度
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()   # 利用gpu训练
            outputs = tudui(imgs)   
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()   # loss整体相加
            accuracy = (outputs.argmax(1) == targets).sum()   # argmax(1)表示横向对比概率选出最大的概率值
            total_accuracy = total_accuracy + accuracy

    # 输出测试结果
    print("整体测试上的loss： {}".format(total_test_loss))
    print("整体测试集上的正确率： {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

     
    # torch.save(tudui, "tudui_{}.pth".format(i))
    # torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))
    # print("模型已保存")

# tensorboard --logdir=logs





