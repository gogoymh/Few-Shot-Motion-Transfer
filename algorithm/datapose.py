import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb
#import cv2
import os
import numpy as np
import random
from torchvision import transforms

#######################################################################################################################
class pre_train(Dataset):
    def __init__(self, path1, path2, path3):
        super().__init__()
        
        self.joint_path = path1
        self.joint_img_path = path2
        self.img_path = path3
        
        self.spatial_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop(256, pad_if_needed=True),
            ])
        
        self.affine_transform = transforms.Compose([
            transforms.RandomAffine(0, shear=[-10, 10, -10, 10]),
            ])
        
        color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        self.color_transform = transforms.Compose([
            transforms.RandomApply([color_jitter], p=0.5)
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = 50678
        
    def __getitem__(self, index):
        
        #joint = imread(os.path.join(self.input_path, "%07d.png" % index))
        #joint_img = imread(os.path.join(self.output_path, "%07d.png" % index))
        #img = imread(os.path.join(self.output_path, "%07d.png" % index))
        
        joint = cv2.imread(os.path.join(self.joint_path, "%07d.png" % index), cv2.IMREAD_COLOR)
        joint_img = cv2.imread(os.path.join(self.joint_img_path, "%07d.png" % index), cv2.IMREAD_COLOR)
        
        ## x1
        seed = np.random.randint(2147483647)
        
        random.seed(seed)
        x1_joint = self.spatial_transform(joint)
        random.seed(seed)
        x1_joint_img = self.spatial_transform(joint_img)
        
        random.seed(seed)
        x1_joint_img1 = self.color_transform(x1_joint_img)
        
        x1_joint_img2 = self.affine_transform(x1_joint_img)
        random.seed(seed)
        x1_joint_img2 = self.color_transform(x1_joint_img2)
        
        x1_joint = self.normalize(x1_joint)
        x1_joint_img1 = self.normalize(x1_joint_img1)
        x1_joint_img2 = self.normalize(x1_joint_img2)
        
        x1 = torch.cat((x1_joint, x1_joint_img1, x1_joint_img2), dim=1)
        
        ## x2
        seed2 = np.random.randint(3214125)
        
        random.seed(seed2)
        x2_joint = self.spatial_transform(joint)
        random.seed(seed2)
        x2_joint_img = self.spatial_transform(joint_img)
        
        x2_joint_img1 = self.affine_transform(x2_joint_img)
        random.seed(seed2)
        x2_joint_img1 = self.color_transform(x2_joint_img1)
        
        x2_joint_img2 = self.affine_transform(x2_joint_img)
        random.seed(seed2)
        x2_joint_img2 = self.color_transform(x2_joint_img2)
        
        x2_joint = self.normalize(x2_joint)
        x2_joint_img1 = self.normalize(x2_joint_img1)
        x2_joint_img2 = self.normalize(x2_joint_img2)
        
        x2 = torch.cat((x2_joint, x2_joint_img1, x2_joint_img2), dim=1)
        
        return x1, x2
    
    def __len__(self):
        return self.len

#######################################################################################################################
class main_train(Dataset):
    def __init__(self, path1, path2, path3):
        super().__init__()
        
        self.joint_path = path1
        self.joint_img_path = path2
        self.img_path = path3
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
            ])
        
        self.spatial_transform0 =transforms.Compose([
            transforms.Resize(512)
            ])
        
        self.spatial_transform1 = transforms.Compose([
            transforms.RandomCrop(350, pad_if_needed=True)
            #transforms.Resize(300)
            ])
        
        self.spatial_transform2 = transforms.Compose([
            transforms.RandomCrop(256, pad_if_needed=True)
            ])
        
        self.affine_transform0 = transforms.Compose([
            transforms.RandomAffine(0, shear=[-8, 8, -8, 8]),
            ])
        
        self.affine_transform1 = transforms.Compose([
            transforms.RandomAffine(0, shear=[-12, 12, -12, 12]),
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = 75650
        
    def __getitem__(self, index):
        
        joint = imread(os.path.join(self.joint_path, "%07d.png" % index))
        joint_img = imread(os.path.join(self.joint_img_path, "%07d_rendered.png" % index))
        img = imread(os.path.join(self.img_path, "%07d.png" % index))
        '''
        
        joint = cv2.imread(os.path.join(self.joint_path, "%07d.png" % index), cv2.IMREAD_COLOR)
        joint_img = cv2.imread(os.path.join(self.joint_img_path, "%07d_rendered.png" % index), cv2.IMREAD_COLOR)
        img = cv2.imread(os.path.join(self.img_path, "%07d.png" % index), cv2.IMREAD_COLOR)
            
        joint = cv2.cvtColor(joint, cv2.COLOR_BGR2RGB)
        joint_img = cv2.cvtColor(joint_img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        '''
        
        '''
        if len(joint.shape) != 3:
            print(index, joint.shape)
            
        if len(joint_img.shape) != 3:
            print(index, joint_img.shape)
            
        if len(img.shape) != 3:
            print(index, img.shape)
        '''
        
        seed = np.random.randint(2147483647)
        
        value = min(joint.shape[0], joint.shape[1])
        
        joint = self.basic_transform(joint)
        joint_img = self.basic_transform(joint_img)
        img = self.basic_transform(img)
        
        if value > 700:
            random.seed(seed)
            joint = self.spatial_transform0(joint)
            random.seed(seed)
            joint_img = self.spatial_transform0(joint_img)
            random.seed(seed)
            img = self.spatial_transform0(img)
        
        if value > 350:
        #if True:
            random.seed(seed)
            joint = self.spatial_transform1(joint)
            random.seed(seed)
            joint_img = self.spatial_transform1(joint_img)
            random.seed(seed)
            img = self.spatial_transform1(img)
        
        input_joint = self.affine_transform0(joint)
        joint_img1 = self.affine_transform1(joint_img)
        joint_img2 = self.affine_transform1(joint_img)
        
        random.seed(seed)
        input_joint = self.spatial_transform2(input_joint)
        random.seed(seed)
        joint = self.spatial_transform2(joint)
        random.seed(seed)
        joint_img1 = self.spatial_transform2(joint_img1)
        random.seed(seed)
        joint_img2 = self.spatial_transform2(joint_img2)
        random.seed(seed)
        img = self.spatial_transform2(img)
        
        input_joint = self.normalize(input_joint)
        joint = self.normalize(joint)
        joint_img1 = self.normalize(joint_img1)
        joint_img2 = self.normalize(joint_img2)
        img = self.normalize(img)
        
        return input_joint, joint_img1, joint_img2, joint, img
    
    def __len__(self):
        return self.len

#######################################################################################################################
class pose_people(Dataset):
    def __init__(self, path1, path2, both_transform=None):#, input_transform=None, output_transform=None):
        super().__init__()
        
        self.input_path = path1
        self.output_path = path2
        
        self.both_transform = both_transform
        #self.input_transform = input_transform
        #self.output_transform = output_transform
        
        self.len = 50678
        
    def __getitem__(self, index):
        
        #input_img = imread(os.path.join(self.input_path, "%07d.png" % index))
        #output_img = imread(os.path.join(self.output_path, "%07d.png" % index))
        
        input_img = cv2.imread(os.path.join(self.input_path, "%07d.png" % index), cv2.IMREAD_COLOR)
        output_img = cv2.imread(os.path.join(self.output_path, "%07d.png" % index), cv2.IMREAD_COLOR)
        
        seed = np.random.randint(2147483647)
        
        if self.both_transform is not None:
            random.seed(seed)
            input_img = self.both_transform(input_img)
            random.seed(seed)
            output_img = self.both_transform(output_img)
        else:
            input_img = input_img.transpose(2,0,1)
            output_img = output_img.transpose(2,0,1)
        
        '''
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        else:
            input_img = input_img.transpose(2,0,1)
            
        if self.output_transform is not None:
            output_img = self.output_transform(output_img)
        else:
            output_img = output_img.transpose(2,0,1)
        '''
        
        return input_img, output_img
    
    def __len__(self):
        return self.len

class people(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        
        self.path = path
        
        self.transform = transform
        
        self.len = 50678
        
    def __getitem__(self, index):
        
        #img = imread(os.path.join(self.path, "%07d.png" % index))
        fname = os.path.join(self.path, "%07d.png" % index)
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
                
        else:
            img1 = img.transpose(2,0,1)
            img2 = img.transpose(2,0,1)
            
        return img1, img2
    
    def __len__(self):
        return self.len

'''
if __name__ == "__main__":
    path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//output"
    
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.ToPILImage(),
         transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
         transforms.RandomCrop(224, pad_if_needed=True),
         transforms.RandomApply([color_jitter], p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    a = people(path, transform)
    train_loader = DataLoader(a, batch_size=32, shuffle=True)#, pin_memory=True)
    print(len(train_loader))
    
    b, c = a.__getitem__(12345)
    print(b.shape)
    print(c.shape)

    import matplotlib.pyplot as plt
    
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


if __name__ == "__main__":
    input_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//input//"
    output_save = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//output//"
    
    from torchvision import transforms
    
    both = transforms.Compose([
        transforms.ToTensor(),
         transforms.ToPILImage(),
         transforms.RandomHorizontalFlip(p=0.5),
         #transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
         transforms.RandomCrop(224, pad_if_needed=True, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    input = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    output = transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    a = pose_people(input_save, output_save, both)#, input, output)
    
    b, c = a.__getitem__(1222)
    print(b.shape)
    print(c.shape)

    import matplotlib.pyplot as plt
    
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

'''


if __name__ == "__main__":
    #path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//input_backup//"
    #path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//input_with_img//"
    #path3 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//output_tmp//"
    
    path1 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//new_input//"
    path2 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//new_input_with_img//"
    path3 = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//new_output_tmp//"
    
    #path1 = "/home/compu/ymh/FSMT/dataset/input/"
    #path2 = "/home/compu/ymh/FSMT/dataset/input_with_img/"
    #path3 = "/home/compu/ymh/FSMT/dataset/output/"
    
    a = main_train(path1, path2, path3)
    '''
    for i in range(75650):
        if i % 1000 == 0 :
            print("[%d/75650]" % i)
        try:
            _, _, _, _, _ = a.__getitem__(i)
        except:
            print(i, "failed")
    '''
    b, c, d, e, f = a.__getitem__(51245)
    
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(e.shape)

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
    
    d[0] = d[0]*0.5 + 0.5
    d[1] = d[1]*0.5 + 0.5
    d[2] = d[2]*0.5 + 0.5
    
    d = d.numpy().transpose(1,2,0)
    
    plt.imshow(d)
    #plt.savefig("/home/compu/ymh/FSMT/dataset/d.png")
    plt.show()
    plt.close()
    
    e[0] = e[0]*0.5 + 0.5
    e[1] = e[1]*0.5 + 0.5
    e[2] = e[2]*0.5 + 0.5
    
    e = e.numpy().transpose(1,2,0)
    
    plt.imshow(e)
    #plt.savefig("/home/compu/ymh/FSMT/dataset/e.png")
    plt.show()
    plt.close()
    
    f[0] = f[0]*0.5 + 0.5
    f[1] = f[1]*0.5 + 0.5
    f[2] = f[2]*0.5 + 0.5
    
    f = f.numpy().transpose(1,2,0)
    
    plt.imshow(f)
    #plt.savefig("/home/compu/ymh/FSMT/dataset/f.png")
    plt.show()
    plt.close()
    












