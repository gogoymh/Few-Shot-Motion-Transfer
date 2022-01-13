import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import numpy as np
from torchvision.utils import save_image
import os

from architecture4 import Generator, Discriminator, Embedder
from dataset2 import pair_set, joint_set

save_path = "/data1/ymh/FSMT/save/new_result9/"

path1 = "/data1/ymh/FSMT/dataset/flip_rendered/"
path2 = "/data1/ymh/FSMT/dataset/flip/"

'''
save_path = "/home/compu/ymh/FSMT/save/new_result7/"

path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered/"
path2 = "/home/compu/ymh/FSMT/dataset/flip/"
'''
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

# Hyper parameter
n_critic = 5
lambda_gp = 10
total = 500000
batch_size = 4
save = 100

# Dataset
pair = pair_set(path1, path2, True)
pair_loader = DataLoader(pair, batch_size=batch_size, shuffle=True)

joint = joint_set(path1, True)
joint_loader = DataLoader(joint, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator().to(device)
#generator = nn.DataParallel(generator)
#generator().to(device)

discriminator = Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

embedder = Embedder().to(device)
#embedder = nn.DataParallel(embedder)
#embedder()

# Optimizers
param = list(generator.parameters()) + list(embedder.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.00005, betas=(0, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00015, betas=(0, 0.9))

fake = Tensor(batch_size, 1).fill_(1.0).to(device)  
for iteration in range(total):
    _, real_imgs = pair_loader.__iter__().next()
    input_joint = joint_loader.__iter__().next()
    
    real_imgs = real_imgs.to(device)
    input_joint = input_joint.to(device)
    
    optimizer_D.zero_grad()
    
    e = embedder(real_imgs)
    fake_imgs = generator(input_joint, e)

    real_validity = discriminator(real_imgs)
    fake_validity = discriminator(fake_imgs)
    
    alpha = Tensor(np.random.random((batch_size, 1, 1, 1))).to(device)
    interpolates = (alpha * real_imgs.data + ((1 - alpha) * fake_imgs.data)).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
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
        
        e = embedder(real_imgs)
        fake_imgs = generator(input_joint, e)
        
        fake_validity = discriminator(fake_imgs)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_G.step()

        print("[Iteration:%d] [D loss: %f] [G loss: %f] [Penelty: %f]" % (iteration, d_loss.item(), g_loss.item(), gradient_penalty.item()))

        if iteration % save == 0:
            save_image(input_joint.data[:4], os.path.join(save_path, "%06d_joint.png" % iteration), nrow=2, normalize=True)
            save_image(fake_imgs.data[:4], os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True)














