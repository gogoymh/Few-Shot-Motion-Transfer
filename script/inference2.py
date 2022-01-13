import torch
from torchvision.utils import save_image
from skimage.io import imread, imsave
import os
from torchvision import transforms
import numpy as np
from new_dataset import reference_set

from new_architecture27 import Generator

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

#Tensor = torch.FloatTensor
#device = torch.device("cpu")

generator = Generator().to(device)

#path = "/home/compu/ymh/FSMT/save/generator56.pth"
path = "/data1/ymh/FSMT/save/generator57.pth"
checkpoint = torch.load(path)

generator.load_state_dict(checkpoint["g"])
generator.eval()

change = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]) 

'''
input_path1 = "/home/compu/ymh/FSMT/dataset/yerin_rendered/"
input_path2 = "/home/compu/ymh/FSMT/dataset/flip/"

save_path = "/home/compu/ymh/FSMT/save/inference56/"
'''

input_path1 = "/data1/ymh/FSMT/dataset/yerin_rendered/"
input_path2 = "/data1/ymh/FSMT/dataset/flip/"

save_path = "/data1/ymh/FSMT/save/inference57/"

reference_from = reference_set(input_path2, 128, True)

#noise = Tensor(np.random.normal(0,1,(1,1,10,6))) #fix noise

for index in range(5940):
    print(index)
    joint = imread(os.path.join(input_path1, "frame_%07d_rendered.png" % index))
    reference = reference_from.__getitem__(0)
    
    joint = change(joint)
    joint = joint.to(device)
    
    reference = reference.to(device)
    
    joint = joint.unsqueeze(0)
    
    #noise = Tensor(np.random.normal(0,1,(1,1,10,6)))
    
    fake = generator(joint, reference)#, noise)
    
    save_image(fake.data[:1], os.path.join(save_path, "frame_%07d.png" % index), nrow=1, normalize=True, range=(-1, 1))
        
    