from torch.utils.data import Dataset
import os
from skimage.io import imread
#import cv2
import numpy as np
import random

class OneShot(Dataset):
    def __init__(self, pose_path, style_path, transform=None):
        super().__init__()
        
        self.pose_path = pose_path
        self.style_path = style_path
        
        self.transform = transform
        
        self.pose_list = os.listdir(pose_path)
        
        self.len = len(self.pose_list)
        
    def __getitem__(self, index):
        pose = imread(os.path.join(self.pose_path, self.pose_list[index]))
        style = imread(self.style_path)
        
        #pose = cv2.imread(os.path.join(self.pose_path, self.pose_list[index]), cv2.IMREAD_COLOR)
        #style = cv2.imread(self.style_path, cv2.IMREAD_COLOR)
        
        seed = np.random.randint(2147483647)
        
        if self.transform is not None:
            random.seed(seed)
            pose = self.transform(pose)
            random.seed(seed)
            style = self.transform(style)
        else:
            pose = pose.transpose(2,0,1)
            style = style.transpose(2,0,1)
        
        return pose, style
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    pose_path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//smp//yerin_save//"
    style_path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//monalisa//monalisa_rendered.png"
    
    from torchvision import transforms
    
    both = transforms.Compose([
         transforms.ToTensor(),
         transforms.ToPILImage(),
         transforms.Resize(512),
         transforms.RandomCrop(256, pad_if_needed=True),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    a = OneShot(pose_path1, style_path1, both)
    
    import matplotlib.pyplot as plt
    
    b, c = a.__getitem__(1000)
    print(b.shape)
    print(c.shape)
    
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
    
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.show()
    plt.close()
    
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
    
    c = c.numpy().transpose(1,2,0)
    
    plt.imshow(c)
    plt.show()
    plt.close()
    
    