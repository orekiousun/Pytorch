from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

#将一张图片转化为tensor类型
image_path="E:\\Courses\\Code\\Python\\Pytorch\\Dataset\\fire\\fire0.jpg"
img=Image.open(image_path)   #打开图片

SummaryWriter

tensor_trans=transforms.ToTensor()   #返回一个totensor对象
tensor_img=tensor_trans(img)   #返回一个tensor类型的图片
print(tensor_img)
# tensorboard --logdir=logs
