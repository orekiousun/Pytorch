from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


writer=SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=4x",4*i,i)

#终端输入：tensorboard --logdir=logs --port=7298  
writer.close()

