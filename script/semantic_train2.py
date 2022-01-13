import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import cv2 as cv
from skimage.io import imread
import os
from torchvision import transforms

from help_set import help_set
from segmentation import Unet, DiceLoss

device = torch.device("cuda:0")

model_name = "/home/compu/ymh/FSMT/save/" + "segmentation_smp4.pth"

path1 = "/home/compu/ymh/FSMT/dataset/video_samples/smp4/"
path2 = "/home/compu/ymh/FSMT/dataset/video_samples/smp4_semantic/"

dataset = help_set(path1, path2, train=True)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

criterion = DiceLoss()

model = Unet().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)


total = 300
for epoch in range(total):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x = x.float().to(device)
        y = y.float().to(device)
            
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
        print(epoch+1, total, i+1, len(train_loader), loss.item())

torch.save({'model_state_dict': model.state_dict()}, model_name)

#checkpoint = torch.load(model_name)
#model.load_state_dict(checkpoint["model_state_dict"])

change = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]) 

model.eval()
for index in range(2500):
    img = imread(os.path.join(path1, "frame_%07d.png" % index))
    x = change(img)
    print(index)
    x = x.unsqueeze(0)
    x = x.float().to(device)
            
    output = model(x)
    pred = (output >= 0.5).float()
        
    pred = pred.squeeze()
    pred = pred.detach().cpu().numpy()
    pred = pred * 255
    pred = pred.astype(np.uint8)
    
    cv.imwrite(os.path.join(path2, "semantic_%07d.png" % index), pred)

#model_name = "/home/compu/ymh/FSMT/save/" + "segmentation.pth"
#torch.save({'model_state_dict': model.state_dict()}, model_name)