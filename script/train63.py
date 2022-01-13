import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture28 import Generator, Discriminator
from new_dataset3 import pair_set



save_path = "/data1/ymh/FSMT/save/new_result63/"
model_name = "/data1/ymh/FSMT/save/" + "generator63.pth"
'''
save_path = "/home/compu/ymh/FSMT/save/new_result61/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator61.pth"
'''
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

n_critic = 5
lambda_gp = 10
total = 800000
batch_size = 23
save = 100


path1 = "/data1/ymh/FSMT/dataset/data/joint_tmp/"
path2 = "/data1/ymh/FSMT/dataset/data/style_tmp/"

'''
path1 = "/home/compu/ymh/FSMT/dataset/data/joint_tmp/"
path2 = "/home/compu/ymh/FSMT/dataset/data/style_tmp/"
'''

pairset = pair_set(path1, path2, True)
pair_loader = DataLoader(pairset, batch_size=batch_size, shuffle=True)


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


checkpoint = torch.load("/data1/ymh/FSMT/save/generator63.pth")
#checkpoint = torch.load("/home/compu/ymh/FSMT/save/generator57.pth")
generator.load_state_dict(checkpoint["g"])
optimizer_G.load_state_dict(checkpoint["g_optim"])

for pre in range(total):
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    
    joint, real = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)
    
    fake = generator(joint, real)

    l1_loss = L1_loss(fake, real) * 100
    
    l1_loss.backward()
    optimizer_G.step()
    print("[Pre iteration:%d] [L1:%f]" % (pre, l1_loss.item()))
    
    if pre % 100 == 0:
        save_image(fake.data[:4], os.path.join(save_path, "%06d_image.png" % pre), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)


for iteration in range(total):
    print("[Iteration:%d]" % iteration, end=" ")
    
    ## ---- D ---- ##
    discriminator.train()
    generator.eval()
    
    for i in range(n_critic):
        optimizer_D.zero_grad()
        joint, real = pair_loader.__iter__().next()
    
        joint = joint.to(device)
        real = real.to(device)
    
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
    
    print("[D:%f] [Penelty:%f]" % (d_loss.item(), gradient_penalty.item()), end=" ")
    
    ## ---- Unrolling ---- ##
    backup = copy.deepcopy(discriminator.state_dict())
    backup2 = copy.deepcopy(optimizer_D.state_dict())
    discriminator.train()
    
    for i in range(n_critic):
        optimizer_D.zero_grad()
        
        real_validity = discriminator(real, joint)
        fake_validity = discriminator(fake.detach(), joint)

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            
        d_loss.backward(create_graph=True)
        optimizer_D.step()
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    discriminator.eval()
    
    fake = generator(joint, real)
    
    fake_validity = discriminator(fake, joint)
    l1_loss = L1_loss(fake, real) * 10
    
    g_loss = -torch.mean(fake_validity) + l1_loss
    g_loss.backward()
    optimizer_G.step()
    print("[G:%f] [L1:%f]" % (g_loss.item(), l1_loss.item()))
    
    discriminator.load_state_dict(backup)    
    del backup
        
    optimizer_D.load_state_dict(backup2)
    del backup2
    
    if iteration % 100 == 0:
        save_image(fake.data[:4], os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)













