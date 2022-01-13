import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import numpy as np
from torchvision.utils import save_image
import os

from architecture3 import Generator1, Discriminator, Embedder, Generator2
from dataset2 import pair_set
'''
save_path = "/data1/ymh/FSMT/save/new_result7/"

path1 = "/data1/ymh/FSMT/dataset/flip2_rendered/"
path2 = "/data1/ymh/FSMT/dataset/flip2/"
'''
save_path = "/home/compu/ymh/FSMT/save/new_result6/"

path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered/"
path2 = "/home/compu/ymh/FSMT/dataset/flip/"

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

# Hyper parameter
n_critic = 5
lambda_gp = 10
lambda_cyc = 20
lambda_id = 5
total = 500000
batch_size = 2
save = 100

# Dataset
pair = pair_set(path1, path2, True)
pair_loader = DataLoader(pair, batch_size=batch_size, shuffle=True)

generator1 = Generator1().to(device)
generator1 = nn.DataParallel(generator1)

discriminator1 = Discriminator().to(device)
discriminator1 = nn.DataParallel(discriminator1)

embedder = Embedder().to(device)
embedder = nn.DataParallel(embedder)

generator2 = Generator2().to(device)
generator2 = nn.DataParallel(generator2)

discriminator2 = Discriminator().to(device)
discriminator2 = nn.DataParallel(discriminator2)

param = list(generator1.parameters()) + list(embedder.parameters())
optimizer_G1 = torch.optim.Adam(param, lr=0.00005, betas=(0, 0.9))
optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=0.00015, betas=(0, 0.9))

optimizer_G2 = torch.optim.Adam(generator2.parameters(), lr=0.00005, betas=(0, 0.9))
optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=0.00015, betas=(0, 0.9))

fake = Tensor(batch_size, 1).fill_(1.0).to(device)

criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

for iteration in range(total):
    joint, real_imgs = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real_imgs = real_imgs.to(device)
    
    ## ---- D1 ---- ##
    optimizer_D1.zero_grad()
    
    y = embedder(real_imgs)
    fake_imgs1 = generator1(joint, y)

    real_validity1 = discriminator1(real_imgs)
    fake_validity1 = discriminator1(fake_imgs1)
    
    alpha = Tensor(np.random.random((batch_size, 1, 1, 1))).to(device)
    interpolates1 = (alpha * real_imgs.data + ((1 - alpha) * fake_imgs1.data)).requires_grad_(True)
    
    d1_interpolates = discriminator1(interpolates1)
    
    gradients1 = autograd.grad(
        outputs=d1_interpolates,
        inputs=interpolates1,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients1 = gradients1.view(gradients1.size(0), -1)
    gradient_penalty1 = ((gradients1.norm(2, dim=1) - 1) ** 2).mean()
    
    d1_loss = -torch.mean(real_validity1) + torch.mean(fake_validity1) + lambda_gp * gradient_penalty1

    d1_loss.backward()
    optimizer_D1.step()
    
    ## ---- D2 ---- ##
    optimizer_D2.zero_grad()
    
    fake_imgs2 = generator2(real_imgs)
    
    real_validity2 = discriminator2(joint)
    fake_validity2 = discriminator2(fake_imgs2)
    
    interpolates2 = (alpha * joint.data + ((1 - alpha) * fake_imgs2.data)).requires_grad_(True)
    
    d2_interpolates = discriminator2(interpolates2)
    
    gradients2 = autograd.grad(
        outputs=d2_interpolates,
        inputs=interpolates2,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients2 = gradients2.view(gradients2.size(0), -1)
    gradient_penalty2 = ((gradients2.norm(2, dim=1) - 1) ** 2).mean()
    
    d2_loss = -torch.mean(real_validity2) + torch.mean(fake_validity2) + lambda_gp * gradient_penalty2

    d2_loss.backward()
    optimizer_D2.step()
    
    optimizer_G1.zero_grad()
    optimizer_G2.zero_grad()
    
    if iteration % n_critic == 0:
        
        y = embedder(real_imgs)
        fake_imgs1 = generator1(joint, y)    
        fake_validity1 = discriminator1(fake_imgs1)
        
        fake_imgs2 = generator2(real_imgs)
        fake_validity2 = discriminator2(fake_imgs2)
        
        adv_loss = (- torch.mean(fake_validity1) - torch.mean(fake_validity2))/2
        
        #loss_id_A = criterion_identity(generator2(joint), joint)
        #loss_id_B = criterion_identity(generator1(real_imgs, y), real_imgs)

        #identity_loss = (loss_id_A + loss_id_B) / 2
        
        recov_A = generator2(fake_imgs1)
        loss_cycle_A = criterion_cycle(recov_A, joint)
        recov_B = generator1(fake_imgs2, y)
        loss_cycle_B = criterion_cycle(recov_B, real_imgs)

        cycle_loss = (loss_cycle_A + loss_cycle_B) / 2
        
        g_loss = adv_loss + lambda_cyc * cycle_loss #+ lambda_id * identity_loss

        g_loss.backward()
        optimizer_G1.step()
        optimizer_G2.step()

        print("[Iteration:%d] [D1: %f] [D2: %f] [G_adv:%f] [G_cyc:%f]" % (iteration, d1_loss.item(), d2_loss.item(), adv_loss.item(), cycle_loss.item()))

        if iteration % save == 0:
            save_image(fake_imgs1.data, os.path.join(save_path, "iter_%06d.png" % iteration), nrow=2, normalize=True)














