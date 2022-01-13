import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture29 import Generator, Discriminator
from new_dataset3 import pair_set
from ContentLoss import LossCnt


save_path = "/data1/ymh/FSMT/save/new_result65/"
model_name = "/data1/ymh/FSMT/save/" + "generator65.pth"
'''
save_path = "/home/compu/ymh/FSMT/save/new_result62/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator62.pth"
'''
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

n_critic = 5
total = 800000
batch_size = 2
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


#L1_loss = nn.L1Loss()
L1_loss = LossCnt(VGGFace_body_path='Pytorch_VGGFACE_IR.py', VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
MSE_loss = nn.MSELoss()

real_value = Tensor(batch_size, 1).fill_(1.0).to(device)
fake_value = Tensor(batch_size*n_critic, 1).fill_(0.0).to(device)

'''
checkpoint = torch.load("/data1/ymh/FSMT/save/generator64.pth")
#checkpoint = torch.load("/home/compu/ymh/FSMT/save/generator57.pth")
generator.load_state_dict(checkpoint["g"])
optimizer_G.load_state_dict(checkpoint["g_optim"])
'''
for pre in range(10):
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    
    joint, real = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)
    
    fake = generator(joint, real)

    l1_loss = L1_loss(fake, real)
    
    l1_loss.backward()
    optimizer_G.step()
    print("[Pre iteration:%d] [L1:%f]" % (pre, l1_loss.item()))
    '''
    if pre % 100 == 0:
        save_image(fake.data[:4], os.path.join(save_path, "%06d_image.png" % pre), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)
    '''

for iteration in range(total):
    fakes = None
    print("[Iteration:%d]" % iteration, end=" ")
    joint, real = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)
    
    ## ---- Unrolling ---- ##
    backup = copy.deepcopy(generator.state_dict())
    backup2 = copy.deepcopy(optimizer_G.state_dict())
    
    generator.train()
    discriminator.eval()
    for i in range(n_critic):
        optimizer_G.zero_grad()            
        fake = generator(joint, real)
        fake_validity = discriminator(fake, joint)
        l1_loss = L1_loss(fake, real)
        g_loss = MSE_loss(fake_validity, real_value) + l1_loss
        g_loss.backward()
        optimizer_G.step()
        
        if fakes == None:
            fakes = fake
        else:
            fakes = torch.cat((fakes, fake), dim=0)
    
    ## ---- D ---- ##
    discriminator.train()
    generator.eval()
    optimizer_D.zero_grad()
        
    real_validity = discriminator(real, joint)
    fake_validity = discriminator(fakes.detach(), joint.repeat((n_critic,1,1,1)))
        
    d_loss = MSE_loss(real_validity, real_value) + (MSE_loss(fake_validity, fake_value))/n_critic

    d_loss.backward()
    optimizer_D.step()
    
    print("[D:%f]" % d_loss.item(), end=" ")
    
    generator.load_state_dict(backup)    
    del backup
        
    optimizer_G.load_state_dict(backup2)
    del backup2
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    discriminator.eval()
    
    fake = generator(joint, real)
    
    fake_validity = discriminator(fake, joint)
    l1_loss = L1_loss(fake, real)
    
    g_loss = MSE_loss(fake_validity, real_value) + l1_loss
    g_loss.backward()
    optimizer_G.step()
    print("[G:%f] [L1:%f]" % (g_loss.item(), l1_loss.item()))
    
    if iteration % 100 == 0:
        save_image(fake.data[:2], os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)













