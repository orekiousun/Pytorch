import torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])   #定义transform功能传入dataset中，转换为tensor类型
train_set = torchvision.datasets.CIFAR10(root="./Dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./Dataset", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)

# img, target = test_set[0]
# print(img, target,test_set.classes[target]) 
# img.show()

# print(test_set[0])   #成功转换为tensor类型

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
# tensorboard --logdir=logs
