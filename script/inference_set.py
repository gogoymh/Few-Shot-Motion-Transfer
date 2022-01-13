import torch
from torch.utils.data import Dataset
import skimage
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms

class reference_set(Dataset):
    def __init__(self, path, num=32, original=True):
        super().__init__()
        
        self.img_path = path
        self.num = num
                
        if original:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192))
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
        
        ref = os.listdir(path)
        self.len = len(ref)
        
    def __getitem__(self, index):
        
        tensor = None
        indices = np.random.choice(self.len, self.num)
        for i in indices:
            img = imread(os.path.join(self.img_path, "frame_%07d.png" % i))
            
            img = self.spatial_transform(img)
            img = self.normalize(img)
            if tensor is None:
                tensor = img.unsqueeze(0)
            else:
                tensor = torch.cat((tensor, img.unsqueeze(0)), dim=0)
        
        return tensor
        
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
        
        self.len = 5940
        
    def __getitem__(self, index):
        
        joint = imread(os.path.join(self.joint_path, "frame_%07d_rendered.png" % index))
        
        joint = self.spatial_transform(joint)
        
        joint = self.normalize(joint)
        
        return joint
        
    def __len__(self):
        return self.len
    

if __name__ == "__main__":
    '''
    path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip//"
    
    a = reference_set(path, 3, True)
    
    b = a.__getitem__(0)
    
    
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip_rendered//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip//"
    path3 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip_semantic4//"
    
    import matplotlib.pyplot as plt
    '''
    
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video_samples//smp4_rendered//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video_samples//smp4//"
    path3 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video_samples//smp4_semantic//"
    
    a = tripple_set(path1, path2, path3, True)
    
    for i in range(2500):
        #print(i)
        b, c, d, e = a.__getitem__(i)
        if e.item() <= 0:
            print(i)
    '''
    import matplotlib.pyplot as plt
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video_samples//smp7_rendered//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video_samples//smp7//"
    path3 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video_samples//smp7_semantic//"
    a = tripple_set(path1, path2, path3, True)
    index = np.random.choice(2500, 1)[0]
    #index = 542
    print(index)
    
    b, c, d, f = a.__getitem__(index)
    
    #print(d.unique())
    
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
    
    d = d.numpy().transpose(1,2,0)
    
    plt.imshow(d, cmap="gray")
    plt.show()
    plt.close()
    
    e = c * d   
    
    e[0] = e[0]*0.5 + 0.5
    e[1] = e[1]*0.5 + 0.5
    e[2] = e[2]*0.5 + 0.5
    
    plt.imshow(e)
    plt.show()
    plt.close()
    
    print(f)
    '''

















