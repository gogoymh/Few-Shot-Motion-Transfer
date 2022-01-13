import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture29 import Generator, Discriminator
from new_dataset4 import pair_set, joint_set
from ContentLoss import LossCnt


save_path = "/data1/ymh/FSMT/save/new_result69/"
model_name = "/data1/ymh/FSMT/save/" + "generator69.pth"
'''
save_path = "/home/compu/ymh/FSMT/save/new_result66/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator66.pth"
'''
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

n_critic = 5
lambda_gp = 10
total = 800000
batch_size = 4
save = 20

'''
path1 = "/home/compu/ymh/FSMT/dataset/data/joint_tmp/"
path2 = "/home/compu/ymh/FSMT/dataset/data/style_tmp/"

'''
path1 = "/data1/ymh/FSMT/dataset/data/joint_tmp/"
path2 = "/data1/ymh/FSMT/dataset/data/style_tmp/"


pairset = pair_set(path1, path2, True)
pair_loader = DataLoader(pairset, batch_size=batch_size, shuffle=True)

jointset = joint_set(path1, True)
joint_loader = DataLoader(jointset, batch_size=batch_size, shuffle=True)

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
#L1_loss = VGGPerceptualLoss().to(device)
Perceptual_loss = LossCnt(VGGFace_body_path='Pytorch_VGGFACE_IR.py', VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)


fake_value = Tensor(batch_size, 1).fill_(1.0).to(device)


#checkpoint = torch.load("/home/compu/ymh/FSMT/save/generator57.pth")
#generator.load_state_dict(checkpoint["g"])
#optimizer_G.load_state_dict(checkpoint["g_optim"])

for pre in range(10):
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    
    joint, real = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)
    
    realistic_fake = generator(joint, real)

    l1_loss = L1_loss(realistic_fake, real) * 100
    
    l1_loss.backward()
    optimizer_G.step()
    print("[Pre iteration:%d] [L1:%f]" % (pre, l1_loss.item()))


for iteration in range(total):
    print("[Iteration:%d]" % iteration, end=" ")
    
    ## ---- D ---- ##
    discriminator.train()
    generator.eval()
    
    for i in range(n_critic):
        optimizer_D.zero_grad()
        joint, real = pair_loader.__iter__().next()
        query = joint_loader.__iter__().next()
    
        joint = joint.to(device)
        real = real.to(device)
        query = query.to(device)
    
        fake = generator(query, real)
        
        real_validity = discriminator(real, joint)
        fake_validity = discriminator(fake.detach(), query)
        
        alpha = Tensor(np.random.random((batch_size, 1, 1, 1)))
        interpolates = (alpha * real.data + ((1 - alpha) * fake.data)).requires_grad_(True)
        joint_interpolates = (alpha * joint.data + ((1 - alpha) * query.data)).requires_grad_(True)
    
        d_interpolates = discriminator(interpolates, joint_interpolates)

        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake_value, create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()
    
    print("[D:%f] [Penelty:%f]" % (d_loss.item(), gradient_penalty.item()), end=" ")
    '''
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
    '''
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    discriminator.eval()
    
    realistic_fake = generator(joint, real)
    fake = generator(query, real)
    
    fake_validity = discriminator(fake, query)
    
    adv_loss = -torch.mean(fake_validity)
    p_loss = Perceptual_loss(fake, real) * 2.5
    l1_loss = L1_loss(realistic_fake, real) * 5
    
    g_loss = adv_loss + p_loss + l1_loss
    g_loss.backward()
    optimizer_G.step()
    print("[G:%f] [P:%f] [L1:%f]" % (g_loss.item(), p_loss.item(), l1_loss.item()))
    '''
    discriminator.load_state_dict(backup)    
    del backup
        
    optimizer_D.load_state_dict(backup2)
    del backup2
    '''
    if iteration % save == 0:
        for_save = torch.cat((fake.data[:2], real.data[:2]), dim=0)
        
        save_image(for_save, os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)













