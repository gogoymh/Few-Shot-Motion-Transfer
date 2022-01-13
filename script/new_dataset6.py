import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb
import os
import numpy as np
import random
from torchvision import transforms

class image_set(Dataset):
    def __init__(self, path1, path2, original=True):
        super().__init__()
        
        self.joint_path = path1
        self.img_path = path2
                
        if original:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192))
                #transforms.Resize(320),
                #transforms.RandomCrop((320,192))
                ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.RandomCrop(256)
                ])        
        
        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.joint_names = os.listdir(self.joint_path)
        self.len = len(self.joint_names)
        
    def __getitem__(self, index):
        
        joint_name = self.joint_names[index]
        
        img_name = joint_name.replace("_rendered", "")
        try:
            img = imread(os.path.join(self.img_path, img_name))
        except:
            img_name = img_name.replace("png", "jpg")
            img = imread(os.path.join(self.img_path, img_name))

        if len(img.shape) == 2:
            img = gray2rgb(img)#, alpha=False)
                    
        img = self.spatial_transform(img)

        img = self.normalize(img)
        
        return img
    
    def __len__(self):
        return self.len

class joint_set(Dataset):
    def __init__(self, path1, original=True):
        super().__init__()
        
        self.joint_path = path1
                
        if original:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192))
                #transforms.Resize(320),
                #transforms.RandomCrop((320,192))
                ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.RandomCrop(256)
                ])        
        
        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.joint_names = os.listdir(self.joint_path)
        self.len = len(self.joint_names)
        
    def __getitem__(self, index):
        
        joint_name = self.joint_names[index]
        joint = imread(os.path.join(self.joint_path, joint_name))
        
        joint = self.spatial_transform(joint)
        
        joint = self.normalize(joint)
        
        return joint
        
    def __len__(self):
        return self.len

class pair_set(Dataset):
    def __init__(self, path1, path2, original=True):
        super().__init__()
        
        self.joint_path = path1
        self.img_path = path2
                
        if original:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192))
                #transforms.Resize(320),
                #transforms.RandomCrop((320,192))
                ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.RandomCrop(256)
                ])        
        
        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.joint_names = os.listdir(self.joint_path)
        self.len = len(self.joint_names)
        
    def __getitem__(self, index):
        
        joint_name = self.joint_names[index]
        joint = imread(os.path.join(self.joint_path, joint_name))
        
        img_name = joint_name.replace("_rendered", "")
        try:
            img = imread(os.path.join(self.img_path, img_name))
        except:
            img_name = img_name.replace("png", "jpg")
            img = imread(os.path.join(self.img_path, img_name))

        if len(img.shape) == 2:
            img = gray2rgb(img)#, alpha=False)
        
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

    import matplotlib.pyplot as plt
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//data//joint_tmp//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//data//style_tmp//"
    
    #path1 = "/home/compu/ymh/FSMT/dataset/data/joint_tmp/"
    #path2 = "/home/compu/ymh/FSMT/dataset/data/style_tmp/"
    
    a = image_set(path1, path2, True)
    index = np.random.choice(43125, 1)[0]
    #index = 0
    print(index)
    
    b = a.__getitem__(index)
    
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
    
    b = b.numpy().transpose(1,2,0)    
    
    plt.imshow(b)
    #plt.savefig("/home/compu/ymh/FSMT/dataset/data/sample_%07d.png" % index)
    plt.show()
    plt.close()   
    '''
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
    
    c = c.numpy().transpose(1,2,0)
       
    plt.imshow(c)
    plt.show()
    plt.close()     
    '''


















