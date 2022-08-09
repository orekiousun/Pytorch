from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("E:\\Courses\\Code\\Python\\Pytorch\\Dataset\\fire\\fire6.jpg")   #PIL格式

# 初始图片
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor, 1)

# 图片归一化
#归一化方式：input[channel] = (input[channel] - mean[channel]) / std[channel]
print("img_tensor[0][0][0]=", img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print("img_norm[0][0][0]=", img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 1)

# Resize：重新设计大小
print("ima_size=", img.size)
trans_resize = transforms.Resize((1200,1600))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)   #img_resize任然是一个PIL类型
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)   #再变化为tensor类型
writer.add_image("Resize", img_resize, 0)

# Compose - resize - 2：重新设计大小
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# Randomcrop 随机裁剪
trans_random = transforms.RandomCrop((500, 1000))   #裁剪结果为500*1000
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])   #表示随机裁剪
for i in range(10):   #裁剪10个
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)



writer.close()
# tensorboard --logdir=logs
