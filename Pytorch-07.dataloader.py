import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("./Dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)   
# shuffle表示十分按照顺序读取，True为乱序读取
# batch_size表示每次从dataset中取4个数据
# drop_last表示是否舍去后面的数据

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
for epcho in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("epoch: {}".format(epcho), imgs , step)
        step = step + 1
writer.close()
# tensorboard --logdir=logs
