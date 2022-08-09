import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn
  
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)   # 制造数据集
y = 3*x + 10 + torch.rand(x.size())   # 引入噪声

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.linear = nn.Linear(1, 1) # 输入和输出的维度都是1

    def forward(self, x):
        output = self.linear(x)
        return output

model = Regression().cuda()

criterion = nn.MSELoss()   # 定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)   # 定义优化器
 
epochs = 500
for epoch in range(epochs):
    print("----------第 {} 次训练开始----------".format(epoch))
    inputs = Variable(x).cuda()
    target = Variable(y).cuda()
 
    # 向前传播
    output = model(inputs)
    loss = criterion(output, target)
 
    # 利用优化器优化
    optimizer.zero_grad() # 梯度清零
    loss.backward()   # 回传
    optimizer.step()

# 测试函数
model.eval()
predict = model(Variable(x).cuda())
predict = predict.data.cpu().numpy()

# 画图
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original Data')
plt.plot(x.numpy(), predict, label='Fitting Line')
plt.show()
