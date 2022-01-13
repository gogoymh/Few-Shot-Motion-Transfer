import torch
from torchvision.utils import save_image
from skimage.io import imread, imsave
import os
from torchvision import transforms
import numpy as np
from inference_set import reference_set, joint_set
from torch.utils.data import DataLoader

from new_architecture27 import Generator

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

#Tensor = torch.FloatTensor
#device = torch.device("cpu")

generator = Generator().to(device)

#path = "/home/compu/ymh/FSMT/save/generator56.pth"
path = "/data1/ymh/FSMT/save/generator58.pth"
checkpoint = torch.load(path)

generator.load_state_dict(checkpoint["g"])
generator.eval()

'''
input_path1 = "/home/compu/ymh/FSMT/dataset/yerin_rendered/"
input_path2 = "/home/compu/ymh/FSMT/dataset/flip/"

save_path = "/home/compu/ymh/FSMT/save/inference56/"
'''

input_path1 = "/data1/ymh/FSMT/dataset/yerin_rendered/"
input_path2 = "/data1/ymh/FSMT/dataset/video_samples/smp2/"

save_path = "/data1/ymh/FSMT/save/inference58/"

joint_from = joint_set(input_path1)
joint_loader = DataLoader(joint_from, batch_size=3, shuffle=False)
reference_from = reference_set(input_path2, 64)

#noise = Tensor(np.random.normal(0,1,(1,1,10,6))) #fix noise

index = 0
for joint in joint_loader:
    reference = reference_from.__getitem__(0)

    joint = joint.to(device)    
    reference = reference.to(device)
    
    #noise = Tensor(np.random.normal(0,1,(1,1,10,6)))
    
    fake = generator(joint, reference)#, noise)
    
    for save_idx in range(3):
        print(index)
        save_image(fake.data[save_idx], os.path.join(save_path, "frame_%07d.png" % index), nrow=1, normalize=True, range=(-1, 1))
        index += 1
        
    
        
    