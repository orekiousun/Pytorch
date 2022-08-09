import torch
import torchvision
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

"""
    loss:<越小越好>
        1.用于计算实际输出和目标之间的差距
        2.为更行输出提供一定的依据
"""

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)
 
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))  

loss_1 = L1Loss()   # 没加reduction表示计算插值之和再除以个数<平均插值>
loss_2 = L1Loss(reduction="sum")   # 加上reduction表示直接计算插值的绝对值之和

result_1 = loss_1(inputs, targets)
result_2 = loss_2(inputs, targets)

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)   # 类似计算方差

print(result_1, result_2, result_mse)

x = torch.tensor([0.1 ,0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))

loss_cross = nn.CrossEntropyLoss()   #交叉商
result_cross = loss_cross(x, y)
print(result_cross)