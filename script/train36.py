import torch
#import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
#from torch.nn import functional as F
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture21 import Generator, Discriminator
from dataset2 import pair_set, joint_set

'''
save_path = "/data1/ymh/FSMT/save/new_result31/"
model_name = "/data1/ymh/FSMT/save/" + "generator31.pth"

path1 = "/data1/ymh/FSMT/dataset/flip2_rendered/"
path2 = "/data1/ymh/FSMT/dataset/flip2/"

'''
save_path = "/home/compu/ymh/FSMT/save/new_result36/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator36.pth"

path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered/"
path2 = "/home/compu/ymh/FSMT/dataset/flip/"

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

print_num = 5
total = 500000
batch_size = 6
save = 100

pair = pair_set(path1, path2, True)
pair_loader = DataLoader(pair, batch_size=batch_size, shuffle=True)

joint = joint_set(path1, True)
joint_loader = DataLoader(joint, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
#generator = nn.DataParallel(generator)
#generator().to(device)

discriminator = Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

param = list(generator.parameters())
optimizer_G = torch.optim.RMSprop(param, lr=0.0001)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)

ones = torch.ones((batch_size, 1)).float().to(device)
zeros = torch.zeros((batch_size, 1)).float().to(device)
for iteration in range(total):
    #######################################################################
    real_joint, real = pair_loader.__iter__().next()
    input_joint = joint_loader.__iter__().next()
    
    input_joint = input_joint.to(device)
    real_joint = real_joint.to(device)
    real = real.to(device)
    
    noise = Tensor(np.random.normal(0,1,(batch_size,1,5,3)))
    
    #######################################################################
    optimizer_D.zero_grad()
    discriminator.train()
    generator.eval()
    
    fake = generator(input_joint, real, noise)
    
    real_validity = discriminator(real, real_joint)
    fake_validity = discriminator(fake.detach(), input_joint)
    
    d_real = 0.5 * torch.mean((real_validity-ones)**2)
    d_fake = 0.5 * torch.mean((fake_validity+ones)**2)
    
    d_loss = d_fake + d_real
        
    d_loss.backward()
    optimizer_D.step()
    
    #######################################################################
    optimizer_G.zero_grad()
    discriminator.eval()
    generator.train()
        
    fake = generator(input_joint, real, noise)
            
    fake_validity = discriminator(fake, input_joint)
    g_loss = 0.5 * torch.mean((fake_validity-zeros)**2)

    g_loss.backward()
    optimizer_G.step()
    
    if iteration % print_num == 0:
        print("[Iteration:%d] [D loss: %f] [G loss: %f]" % (iteration, d_loss.item(), g_loss.item()))
        
    if iteration % save == 0:
        save_image(input_joint.data[:4], os.path.join(save_path, "%06d_1input.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        save_image(fake.data[:4], os.path.join(save_path, "%06d_2generated.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        save_image(real.data[:4], os.path.join(save_path, "%06d_3style.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
            
        torch.save({'model_state_dict': generator.state_dict()}, model_name)














