"""
    1.加载数据
    Dataset:提供一种方式去获取数据及其label
        如何获取每一个数据机器label
        告诉我们数据总数
    Dataloder:为后面的网络提供不同的数据形式
"""
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):   #构造函数
        self.root_dir=root_dir   
        self.label_dir=label_dir   #初始化
        self.path=os.path.join(self.root_dir,self.label_dir)   #将路径拼接
        self.img_path=os.listdir(self.path)   #创建文件列表，将每个图片作为列表放入img_path中

    def __getitem__(self,idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)   #输出相对路径
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)
    
root_dir="dataset/train"
name_1_label_dir="name_1"
name_2_label_dir="name_2"
name_1_dataset=MyData(root_dir,name_1_label_dir)   #此时name_dataset[x]可以直接索引
name_2_dataset=MyData(root_dir,name_2_label_dir)
train_dataset=name_1_dataset+name_2_dataset






