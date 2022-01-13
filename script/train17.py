import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
from torch.nn import functional as F
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture7 import Generator, Discriminator
from dataset2 import pair_set


save_path = "/data1/ymh/FSMT/save/new_result17/"

path1 = "/data1/ymh/FSMT/dataset/flip_rendered_cp/"
path2 = "/data1/ymh/FSMT/dataset/flip_cp/"

'''
save_path = "/home/compu/ymh/FSMT/save/new_result16/"

path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered_cp/"
path2 = "/home/compu/ymh/FSMT/dataset/flip_cp/"
'''
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

n_critic = 5
lambda_gp = 10
total = 500000
batch_size = 4
save = 100

pair = pair_set(path1, path2, True)
pair_loader = DataLoader(pair, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
#generator = nn.DataParallel(generator)
#generator().to(device)

discriminator = Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

param = list(generator.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.0001, betas=(0, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0, 0.9))

fake_value = Tensor(batch_size, 1).fill_(1.0).to(device)  
for iteration in range(total):
    joint, real = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)
    
    optimizer_D.zero_grad()
    discriminator.train()
    generator.eval()
    
    fake = generator(joint, real)

    real_validity = discriminator(real, joint)
    fake_validity = discriminator(fake.detach(), joint)
    
    alpha = Tensor(np.random.random((batch_size, 1, 1, 1)))
    interpolates = (alpha * real.data + ((1 - alpha) * fake.data)).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates, joint)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_value,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

    d_loss.backward()
    optimizer_D.step()
    
    optimizer_G.zero_grad()
    
    if iteration % n_critic == 0:
        discriminator.eval()
        generator.train()
        
        fake = generator(joint, real)
        
        fake_validity = discriminator(fake, joint)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_G.step()

        print("[Iteration:%d] [D loss: %f] [G loss: %f] [Penelty: %f]" % (iteration, d_loss.item(), g_loss.item(), gradient_penalty.item()))

        if iteration % save == 0:
            save_image(joint.data[:4], os.path.join(save_path, "%06d_joint.png" % iteration), nrow=2, normalize=True)
            save_image(fake.data[:4], os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True)














