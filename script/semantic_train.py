import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import cv2 as cv

from semantic_set import semantic_set
from segmentation import Unet, DiceLoss

device = torch.device("cuda:0")


path1 = "/home/compu/ymh/FSMT/dataset/flip/"
path2 = "/home/compu/ymh/FSMT/dataset/flip_semantic/"

dataset = semantic_set(path1, path2, train=False)
dataset2 = semantic_set(path1, path2, train=False)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset2, batch_size=1, shuffle=False)

criterion = DiceLoss()

total = 4
for train in range(10):
    total += 1
    
    model = Unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

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
    
            print(train, epoch, total, i, len(train_loader), loss.item())
    
        if epoch >= 2:
            model.eval()
            for index, (x, y) in enumerate(valid_loader):
                print(index)
                x = x.float().to(device)
                y = y.float().to(device)
            
                output = model(x)
                pred = (output >= 0.5).float()
        
                pred = 0.8*y + 0.2*pred
                
                pred = pred.squeeze()
                pred = pred.detach().cpu().numpy()
                pred = pred * 255
                pred = pred.astype(np.uint8)
    
                cv.imwrite("/home/compu/ymh/FSMT/dataset/flip_semantic/semantic_%07d.png" % index, pred)
    