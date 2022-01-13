import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os
from torchvision import transforms

from new_architecture35 import Discriminator, Generator
from new_dataset5 import joint_set

'''
save_path = "/data1/ymh/FSMT/save/new_result70/"
model_name = "/data1/ymh/FSMT/save/" + "generator70.pth"
'''
save_path = "/home/compu/ymh/FSMT/save/new_result73/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator73.pth"

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

n_critic = 5
lambda_gp = 10
total = 800000
batch_size = 10
save = 20

path1 = "/home/compu/ymh/FSMT/dataset/Face/"

imageset = joint_set(path1, True)
image_loader = DataLoader(imageset, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
#generator = nn.DataParallel(generator)
#generator().to(device)

discriminator = Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

param = list(generator.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.0001, betas=(0, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

fake_value = Tensor(batch_size, 1).fill_(1.0).to(device)
L1_loss = nn.L1Loss()

#checkpoint = torch.load("/home/compu/ymh/FSMT/save/generator57.pth")
#generator.load_state_dict(checkpoint["g"])
#optimizer_G.load_state_dict(checkpoint["g_optim"])

'''
for pre in range(10):
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    
    real = image_loader.__iter__().next()
    real = real.to(device)
    noise = Tensor(np.random.normal(0, 1, (batch_size, 512)))
    noise = noise.to(device)
    
    fake = generator(noise)

    l1_loss = L1_loss(fake, real) * 100
    
    l1_loss.backward()
    optimizer_G.step()
    print("[Pre iteration:%d] [L1:%f]" % (pre, l1_loss.item()))
'''

for iteration in range(total):
    print("[Iteration:%d]" % iteration, end=" ")
    
    ## ---- D ---- ##
    discriminator.train()
    generator.eval()
    
    for i in range(n_critic):
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        
        real = image_loader.__iter__().next()
        real = real.to(device)
        noise = Tensor(np.random.normal(0, 1, (batch_size, 512)))
        noise = noise.to(device)
        
        fake = generator(noise)
        fake = fake.detach()
        
        real_validity = discriminator(real)
        fake_validity = discriminator(fake)
        
        alpha = Tensor(np.random.random((batch_size, 1, 1, 1)))
        interpolates = (alpha * real.data + ((1 - alpha) * fake.data)).requires_grad_(True)
    
        d_interpolates = discriminator(interpolates)

        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake_value, create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        a = real_validity.mean()
        b = fake_validity.mean()
        c = lambda_gp * gradient_penalty
        
        d_loss = -a + b + c 

        d_loss.backward()
        optimizer_D.step()
    
    print("[D real:%f] [D fake:%f] [Penalty:%f]" % (a.item(), b.item(), c.item()), end=" ") 
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    discriminator.eval()
    
    fake = generator(noise)
    fake_validity = discriminator(fake)
    
    g_loss = -fake_validity.mean()
    g_loss.backward()
    optimizer_G.step()
    print("[G:%f]" % g_loss.item())

    if iteration % save == 0:
        save_image(fake.data[:4], os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)













