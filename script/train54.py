import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture27 import Generator, Discriminator
from new_dataset import tripple_set, reference_set

'''
save_path = "/data1/ymh/FSMT/save/new_result42/"
model_name = "/data1/ymh/FSMT/save/" + "generator42.pth"

path1 = "/data1/ymh/FSMT/dataset/flip_rendered/"
path2 = "/data1/ymh/FSMT/dataset/flip/"

'''
save_path = "/home/compu/ymh/FSMT/save/new_result54/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator54.pth"

path1 = "/home/compu/ymh/FSMT/dataset/flip_rendered/"
path2 = "/home/compu/ymh/FSMT/dataset/flip/"
path3 = "/home/compu/ymh/FSMT/dataset/flip_semantic_re/"

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

num_ref = 8
n_critic = 5
lambda_gp = 10
total = 800000
batch_size = 2
save = 100

tripple = tripple_set(path1, path2, path3, True)
tripple_loader = DataLoader(tripple, batch_size=batch_size, shuffle=True)
#pre_tripple_loader = DataLoader(tripple, batch_size=14, shuffle=True)

reference_from = reference_set(path2, (num_ref-batch_size), True)
#pre_reference_from = reference_set(path2, 2, True)

generator = Generator().to(device)
#generator = nn.DataParallel(generator)
#generator().to(device)

discriminator = Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

param = list(generator.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.0001, betas=(0, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

#L1_loss = nn.L1Loss()

fake_value = Tensor(batch_size, 1).fill_(1.0).to(device)

checkpoint = torch.load("/home/compu/ymh/FSMT/save/generator53.pth")
generator.load_state_dict(checkpoint["g"])
#optimizer_G.load_state_dict(checkpoint["g_optim"])

'''
for pre_iteration in range(100000):
    print("[Pre iteration:%d]" % pre_iteration, end=" ")
    joint, real, semantic, mask_num = pre_tripple_loader.__iter__().next()
    reference = pre_reference_from.__getitem__(0)
    
    joint = joint.to(device)
    real = real.to(device)
    semantic = semantic.to(device)
    mask_num = mask_num.to(device)
    mask_num = mask_num.reshape(14,1)
    noise = Tensor(np.random.normal(0,1,(14,1,5,3)))
    
    reference = reference.to(device)
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    
    style = torch.cat((real,reference), dim=0)
    shuffle = torch.randperm(16)
    fake = generator(joint, style[shuffle], noise)
    
    l1_loss = (10 * torch.abs((fake * semantic - real * semantic))).reshape(14,-1)
    l1_loss = l1_loss.sum(dim=1, keepdim=True)
    l1_loss = l1_loss/mask_num
    l1_loss = l1_loss.mean()

    l1_loss.backward()
    optimizer_G.step()
    print("[L1 loss:%f]" % l1_loss.item())
    
    if pre_iteration % 100 == 0:
        save_image(fake.data[:1], os.path.join(save_path, "%06d_pre_image.png" % pre_iteration), nrow=1, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)
'''

for iteration in range(total):
    print("[Iteration:%d]" % iteration, end=" ")
    joint, real, semantic, mask_num = tripple_loader.__iter__().next()
    reference = reference_from.__getitem__(0)
    
    joint = joint.to(device)
    real = real.to(device)
    semantic = semantic.to(device)
    mask_num = mask_num.to(device)
    mask_num = mask_num.reshape(batch_size,1)
    
    reference = reference.to(device)
    
    style = torch.cat((real,reference), dim=0)
    shuffle = torch.randperm(num_ref)
    
    '''
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    discriminator.eval()
    generator.train()
    
    style = torch.cat((real,reference), dim=0)
    shuffle = torch.randperm(num_ref)
    fake = generator(joint, style[shuffle], noise)
    
    l1_loss = (10 * torch.abs((fake * semantic - real * semantic))).reshape(batch_size,-1)
    l1_loss = l1_loss.sum(dim=1, keepdim=True)
    l1_loss = l1_loss/mask_num
    l1_loss = l1_loss.mean()
    
    
    if l1_loss.item() >= 1:
        l1_loss.backward()
        optimizer_G.step()
        print("[L1 loss:%f]" % l1_loss.item())
        
    else:
    '''
    ## ---- D ---- ##
    optimizer_D.zero_grad()
    discriminator.train()
    generator.eval()
        
    fake = generator(joint, style[shuffle])
        
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
    backup2 = copy.deepcopy(optimizer_D.state_dict())
    
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
        
    fake = generator(joint, style[shuffle])
        
    l1_loss = (200 * torch.abs((fake * semantic - real * semantic))).reshape(batch_size,-1)
    l1_loss = l1_loss.sum(dim=1, keepdim=True)
    l1_loss = l1_loss/mask_num
    l1_loss = l1_loss.mean()
        
    fake_validity = discriminator(fake, joint)
    g_loss = -torch.mean(fake_validity) + l1_loss
    g_loss.backward()
    optimizer_G.step()
    print("[g loss:%f]" % g_loss.item())
    
    discriminator.load_state_dict(backup)    
    del backup
        
    optimizer_D.load_state_dict(backup2)
    del backup2

    if iteration % save == 0:
        #save_image(joint.data[:4], os.path.join(save_path, "%06d_joint.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        save_image(fake.data[:1], os.path.join(save_path, "%06d_image.png" % iteration), nrow=1, normalize=True, range=(-1, 1))
            
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)
    if iteration % 1000 == 0 :
        n_critic += 1













