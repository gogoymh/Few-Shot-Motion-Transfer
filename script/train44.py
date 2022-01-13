import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture13 import Generator, Discriminator
from new_dataset import tripple_set

'''
save_path = "/data1/ymh/FSMT/save/new_result42/"
model_name = "/data1/ymh/FSMT/save/" + "generator42.pth"

path1 = "/data1/ymh/FSMT/dataset/flip_rendered/"
path2 = "/data1/ymh/FSMT/dataset/flip/"

'''
save_path = "/home/compu/ymh/FSMT/save/new_result44/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator44.pth"

path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered_cp/"
path2 = "/home/compu/ymh/FSMT/dataset/flip_cp/"
path3 = "/home/compu/ymh/FSMT/dataset/flip_semantic_re/"

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

n_critic = 10
lambda_gp = 10
total = 500000
batch_size = 2
save = 100

tripple = tripple_set(path1, path2, path3, True)
tripple_loader = DataLoader(tripple, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
#generator = nn.DataParallel(generator)
#generator().to(device)

discriminator = Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

param = list(generator.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.0001, betas=(0, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

L1_loss = nn.L1Loss()

fake_value = Tensor(batch_size, 1).fill_(1.0).to(device)  
for iteration in range(total):
    print("[Iteration:%d]" % iteration, end=" ")
    joint, real, semantic = tripple_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)
    semantic = semantic.to(device)
    noise = Tensor(np.random.normal(0,1,(batch_size,1,10,6)))
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    discriminator.eval()
    generator.train()
    
    fake = generator(joint, real, noise)
    l1_loss_a = 100 * L1_loss(fake * semantic, real * semantic)
    l1_loss_b = 10 * L1_loss(fake, real)
    
    if l1_loss_a.item() >= 1 or l1_loss_b.item() >= 4:
        l1_loss = l1_loss_a + l1_loss_b
        l1_loss.backward()
        optimizer_G.step()
        print("[L1 loss:%f]" % l1_loss.item())
        
    else:
        ## ---- D ---- ##
        optimizer_D.zero_grad()
        discriminator.train()
        generator.eval()
        
        fake = generator(joint, real, noise)
        
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
    
        print("[D loss: %f] [Penelty: %f]" % (d_loss.item(), gradient_penalty.item()), end=" ")
    
        ## ---- Unrolling ---- ##
        backup = copy.deepcopy(discriminator.state_dict())
    
        for i in range(n_critic):
            optimizer_D.zero_grad()
            discriminator.train()
            generator.eval()

            real_validity = discriminator(real, joint)
            fake_validity = discriminator(fake.detach(), joint)
            '''
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
            '''
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)# + lambda_gp * gradient_penalty
            
            d_loss.backward(create_graph=True)
            optimizer_D.step()
        
        ## ---- G ---- ##
        optimizer_G.zero_grad()
        discriminator.eval()
        generator.train()
        
        fake = generator(joint, real, noise)
        l1_loss = 100 * L1_loss(fake * semantic, real * semantic) + 10 * L1_loss(fake, real)
        
        fake_validity = discriminator(fake, joint)
        g_loss = -torch.mean(fake_validity) + l1_loss
        g_loss.backward()
        optimizer_G.step()
        print("[g loss:%f]" % g_loss.item())
    
        discriminator.load_state_dict(backup)    
        del backup

    if iteration % save == 0:
        #save_image(joint.data[:4], os.path.join(save_path, "%06d_joint.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        save_image(fake.data[:2], os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
            
        torch.save({'model_state_dict': generator.state_dict()}, model_name)














