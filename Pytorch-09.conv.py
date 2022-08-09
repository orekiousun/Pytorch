import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1], 
                      [1, 2, 1, 0, 0], 
                      [5, 2, 3, 1, 1], 
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))   # 改变维度
kernel = torch.reshape(kernel, (1, 1, 3, 3))   
print(input.shape)
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)   # stride表示每次每次移动一个路径，stride为2就每次移动两个路径
print(output1)

output2 = F.conv2d(input, kernel, stride=2)   # stride表示每次每次移动一个路径，stride为2就每次移动两个路径
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)   #padding表示在外部进行0扩展之后进行卷积
print(output3)

