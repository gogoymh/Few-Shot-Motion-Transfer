import torch
from torchvision.utils import save_image
from skimage.io import imread, imsave
import os
from torchvision import transforms
import numpy as np

from new_architecture13 import Generator

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

#Tensor = torch.FloatTensor
#device = torch.device("cpu")

generator = Generator().to(device)

path = "/home/compu/ymh/FSMT/save/generator44.pth"
checkpoint = torch.load(path)

generator.load_state_dict(checkpoint["model_state_dict"])
generator.eval()

change = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]) 

input_path1 = "/home/compu/ymh/FSMT/dataset/yerin_rendered/"
input_path2 = "/home/compu/ymh/FSMT/dataset/flip_cp/"

save_path = "/home/compu/ymh/FSMT/save/inference44/"

for index in range(5940):
    print(index)
    joint = imread(os.path.join(input_path1, "frame_%07d_rendered.png" % index))
    #style_index = np.random.choice(5856, 1)[0]
    style_index = 4334
    style = imread(os.path.join(input_path2, "%07d.png" % style_index))
    
    joint = change(joint)
    joint = joint.to(device)
    
    style = change(style)
    style = style.to(device)
    
    joint = joint.unsqueeze(0)
    style = style.unsqueeze(0)
    
    noise = Tensor(np.random.normal(0,1,(1,1,10,6)))
    
    fake = generator(joint, style, noise)
    
    save_image(fake.data[:1], os.path.join(save_path, "frame_%07d.png" % index), nrow=1, normalize=True, range=(-1, 1))
        
    