import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms

class joint_set(Dataset):
    def __init__(self, path1, original=False):
        super().__init__()
        
        self.joint_path = path1
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
            ])
        
        if original:
            self.spatial_transform = transforms.Compose([
            transforms.Resize((320,192))
            ])
        
        else:
            self.spatial_transform = transforms.Compose([
            transforms.RandomCrop(256)
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = 34581
        
    def __getitem__(self, index):
        
        joint = imread(os.path.join(self.joint_path, "%07d_rendered.png" % index))
        
        joint = self.basic_transform(joint)
        
        joint = self.spatial_transform(joint)
        
        joint = self.normalize(joint)
        
        return joint
        
    def __len__(self):
        return self.len

class pair_set(Dataset):
    def __init__(self, path1, path2, original=False):
        super().__init__()
        
        self.joint_path = path1
        self.img_path = path2
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
            ])
        
        if original:
            self.spatial_transform = transforms.Compose([
            transforms.Resize((320,192))
            ])
        else:
            self.spatial_transform = transforms.Compose([
            transforms.RandomCrop(256)
            ])        
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = 34581
        
    def __getitem__(self, index):
        
        joint = imread(os.path.join(self.joint_path, "%07d_rendered.png" % index))
        img = imread(os.path.join(self.img_path, "%07d.png" % index))
        
        joint = self.basic_transform(joint)
        img = self.basic_transform(img)
        
        seed = np.random.randint(2147483647)
        
        random.seed(seed)
        joint = self.spatial_transform(joint)
        random.seed(seed)
        img = self.spatial_transform(img)
        
        joint = self.normalize(joint)
        img = self.normalize(img)
        
        return joint, img
        
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//smp_result//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//video_save//"
    
    a = pair_set(path1, path2, True)
    
    index = np.random.choice(3000)
    
    b, c = a.__getitem__(index)
    
    print(b.shape)
    print(c.shape)

    import matplotlib.pyplot as plt
    
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
    
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    #plt.savefig("/home/compu/ymh/FSMT/dataset/b.png")
    plt.show()
    plt.close()
    
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
    
    c = c.numpy().transpose(1,2,0)
    
    plt.imshow(c)
    #plt.savefig("/home/compu/ymh/FSMT/dataset/c.png")
    plt.show()
    plt.close()
    '''
    
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//smp_result//"
    
    a = joint_set(path1)
    
    b = a.__getitem__(0)
    
    print(b.shape)

    import matplotlib.pyplot as plt
    
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
    
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    #plt.savefig("/home/compu/ymh/FSMT/dataset/b.png")
    plt.show()
    plt.close()
    '''
    