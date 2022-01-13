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
        
        self.len = 5856
        
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
        
        self.len = 5856
        
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
    
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip_rendered//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip//"
    
    import matplotlib.pyplot as plt
    
    a = pair_set(path1, path2, True)
    
    index = np.random.choice(5856, 1)
    index = 3239
    print(index)
    
    b, c = a.__getitem__(index)
    
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
       
    
    d = c.numpy().transpose(1,2,0)
    
    plt.imshow(d)
    plt.show()
    plt.close()
    
    e1 = (c[0,:,:] >= (163/255)).float()
    e2 = (c[0,:,:] <= (173/255)).float()
    e3 = (c[1,:,:] >= (130/255)).float()
    e4 = (c[1,:,:] <= (145/255)).float()
    e5 = (c[2,:,:] >= (130/255)).float()
    e6 = (c[2,:,:] <= (145/255)).float()
    e = ((e1 + e2 + e3 + e4 + e5 + e6) == 6).float()
    
    f1 = (c[0,:,:] >= (175/255)).float()
    f2 = (c[0,:,:] <= (185/255)).float()
    f3 = (c[1,:,:] >= (145/255)).float()
    f4 = (c[1,:,:] <= (165/255)).float()
    f5 = (c[2,:,:] >= (150/255)).float()
    f6 = (c[2,:,:] <= (170/255)).float()
    f = ((f1 + f2 + f3 + f4 + f5 + f6) == 6).float()
    
    g1 = (c[0,:,:] >= (115/255)).float()
    g2 = (c[0,:,:] <= (135/255)).float()
    g3 = (c[1,:,:] >= (70/255)).float()
    g4 = (c[1,:,:] <= (95/255)).float()
    g5 = (c[2,:,:] >= (70/255)).float()
    g6 = (c[2,:,:] <= (95/255)).float()
    g = ((g1 + g2 + g3 + g4 + g5 + g6) == 6).float()
       
    j1 = (c[0,:,:] >= (190/255)).float()
    j2 = (c[0,:,:] <= (220/255)).float()
    j3 = (c[1,:,:] >= (145/255)).float()
    j4 = (c[1,:,:] <= (195/255)).float()
    j5 = (c[2,:,:] >= (145/255)).float()
    j6 = (c[2,:,:] <= (200/255)).float()
    j = ((j1 + j2 + j3 + j4 + j5 + j6) == 6).float()
        
    n1 = (c[0,:,:] >= (217/255)).float()
    n2 = (c[0,:,:] <= (240/255)).float()
    n3 = (c[1,:,:] >= (200/255)).float()
    n4 = (c[1,:,:] <= (230/255)).float()
    n5 = (c[2,:,:] >= (190/255)).float()
    n6 = (c[2,:,:] <= (230/255)).float()
    n = ((n1 + n2 + n3 + n4 + n5 + n6) == 6).float()
    
    k1 = (c[0,:,:] >= (145/255)).float()
    k2 = (c[0,:,:] <= (150/255)).float()
    k3 = (c[1,:,:] >= (90/255)).float()
    k4 = (c[1,:,:] <= (110/255)).float()
    k5 = (c[2,:,:] >= (90/255)).float()
    k6 = (c[2,:,:] <= (135/255)).float()
    k = ((k1 + k2 + k3 + k4 + k5 + k6) == 6).float()
    
    h1 = (c[0,:,:] >= (160/255)).float()
    h2 = (c[0,:,:] <= (170/255)).float()
    h3 = (c[1,:,:] >= (120/255)).float()
    h4 = (c[1,:,:] <= (140/255)).float()
    h5 = (c[2,:,:] >= (120/255)).float()
    h6 = (c[2,:,:] <= (130/255)).float()
    h = ((h1 + h2 + h3 + h4 + h5 + h6) == 6).float()
        
    boundary = j + n + f + g + k + e + h
    boundary = (boundary >= 0.5).float()
    boundary = boundary.numpy()

    #boundary[270:,:] = 0
    
    plt.imshow(boundary, cmap = "gray")
    plt.show()
    plt.close()
    
    #if d[:,:,2] >= 85 and d[:,:,2] =< 95:
        
    
    '''
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
    