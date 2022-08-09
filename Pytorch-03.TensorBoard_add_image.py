from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import torchvision

writer=SummaryWriter("logs")
image_path="E:\\Courses\\AI\\4th work\\Dataset\\fire\\fire2.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(type(img_array))
print((img_array.shape))
writer.add_image("test", img_array,1,dataformats='HWC')
writer.close()
# tensorboard --logdir=logs