import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import numpy as np
from torchvision.utils import save_image
import os

from architecture2 import Generator, Discriminator, Embedder
from dataset2 import pair_set, joint_set

save_path = "/data1/ymh/FSMT/save/new_result3/"

path1 = "/data1/ymh/FSMT/dataset/flip_rendered/"
path2 = "/data1/ymh/FSMT/dataset/flip/"

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

# Hyper parameter
n_critic = 5
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
generator = nn.DataParallel(generator)
#generator().to(device)

discriminator = Discriminator().to(device)
discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

embedder = Embedder().to(device)
embedder = nn.DataParallel(embedder)
#embedder()

# Optimizers
param = list(generator.parameters()) + list(embedder.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.00005, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00015, betas=(0.5, 0.999))

adversarial_loss = torch.nn.BCELoss()

valid = Tensor(batch_size, 1).fill_(1.0)
fake = Tensor(batch_size, 1).fill_(0.0)

for iteration in range(total):
    joint, real_imgs = pair_loader.__iter__().next()
    input_joint = joint_loader.__iter__().next()
    
    joint = joint.to(device)
    real_imgs = real_imgs.to(device)
    input_joint = input_joint.to(device)
    
    optimizer_G.zero_grad()
    
    e = embedder(real_imgs)
    fake_imgs = generator(input_joint, e)
    
    g_loss = adversarial_loss(discriminator(fake_imgs, input_joint), valid)

    g_loss.backward()
    optimizer_G.step()
    
    optimizer_D.zero_grad()    
   
    real_loss = adversarial_loss(discriminator(real_imgs, joint), valid)
    fake_loss = adversarial_loss(discriminator(fake_imgs.detach(), input_joint), fake)
    
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()
    
    if iteration % n_critic == 0:
        print("[Iteration:%d] [D loss: %f] [G loss: %f]" % (iteration, d_loss.item(), g_loss.item()))

    if iteration % save == 0:
            save_image(fake_imgs.data[:4], os.path.join(save_path, "iter_%06d.png" % iteration), nrow=2, normalize=True)














