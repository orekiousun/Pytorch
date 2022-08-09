import torch
import torchvision

# 加载模型<对应保存方式一>
model = torch.load("vgg16_method1.pth")
print(model)

# 加载模型<对应保存方式二>
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)