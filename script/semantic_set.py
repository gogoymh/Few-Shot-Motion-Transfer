import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms

class semantic_set(Dataset):
    def __init__(self, path1, path2, train=True):
        super().__init__()
        
        self.img_path = path1
        self.segmantic_path = path2
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        if train:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192)),
                RandomHorizontalFlip(),
                transforms.RandomAffine(0, shear=[-15, 15, -15, 15])
                ])  
        else:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192))
                ])  
        
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = 5856
        
    def __getitem__(self, index):
        
        img = imread(os.path.join(self.img_path, "frame_%07d.png" % index))
        semantic = imread(os.path.join(self.segmantic_path, "semantic_%07d.png" % index), as_gray=True)
        semantic = semantic * 255
        semantic = semantic.astype(np.uint8)
        
        seed = np.random.randint(2147483647)
        
        random.seed(seed)
        img = self.spatial_transform(img)
        random.seed(seed)
        semantic = self.spatial_transform(semantic)
        
        semantic = self.basic_transform(semantic)
        img = self.normalize(img)
        
        return img, semantic
        
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//video//flip_semantic4//"
    
    import matplotlib.pyplot as plt
    
    a = semantic_set(path1, path2,False)
    
    index = np.random.choice(5856, 1)
    index = 5774
    print(index)
    
    b, c = a.__getitem__(index)
    #print(b)
    #print(c)
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.show()
    plt.close()
    
    c = c.squeeze().numpy()
    
    plt.imshow(c, cmap="gray")
    plt.show()
    plt.close()
    
    
    
    
    
    
    
    
    