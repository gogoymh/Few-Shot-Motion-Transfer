import torch
from torch.utils.data import Dataset
import skimage
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms

class reference_set(Dataset):
    def __init__(self, path, num=3, original=False):
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
        
        self.len = 5856
        
    def __getitem__(self, index):
        
        tensor = None
        indices = np.random.choice(5856, self.num)
        for i in indices:
            img = imread(os.path.join(self.img_path, "%07d.png" % i))
            
            img = self.spatial_transform(img)
            img = self.normalize(img)
            if tensor is None:
                tensor = img.unsqueeze(0)
            else:
                tensor = torch.cat((tensor, img.unsqueeze(0)), dim=0)
        
        return tensor
        
    def __len__(self):
        return self.len

class tripple_set(Dataset):
    def __init__(self, path1, path2, path3, original=False):
        super().__init__()
        
        self.joint_path = path1
        self.img_path = path2
        self.segmantic_path = path3
                
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
        
        self.len = 5856
        
    def __getitem__(self, index):
        
        joint = imread(os.path.join(self.joint_path, "%07d_rendered.png" % index))
        img = imread(os.path.join(self.img_path, "%07d.png" % index))
        semantic = imread(os.path.join(self.segmantic_path, "semantic_%07d.png" % index))
        semantic = skimage.color.gray2rgb(semantic, alpha=False)
        
        if semantic.shape[2] != 3:
            print(index)
        
        seed = np.random.randint(2147483647)
        
        random.seed(seed)
        joint = self.spatial_transform(joint)
        random.seed(seed)
        img = self.spatial_transform(img)
        random.seed(seed)
        semantic = self.spatial_transform(semantic)
        
        joint = self.normalize(joint)
        img = self.normalize(img)
        semantic = self.basic_transform(semantic)
        
        mask_num = (semantic==1).sum()
        
        return joint, img, semantic, mask_num
        
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
    '''
    path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered/"
    path2 = "/home/compu/ymh/FSMT/dataset/flip/"
    path3 = "/home/compu/ymh/FSMT/dataset/flip_semantic_re/"
    
    a = tripple_set(path1, path2, path3, True)
    
    for i in range(5856):
        #print(i)
        _ = a.__getitem__(i)
    
    '''
    import matplotlib.pyplot as plt
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip_rendered//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip//"
    path3 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip_semantic4//"
    a = tripple_set(path1, path2, path3, True)
    index = np.random.choice(5856, 1)[0]
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


















