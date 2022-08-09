import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False) 

# 保存方式一，通过方式一自己定义模型有 
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式二，将vgg16的参数保存为字典形式
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
