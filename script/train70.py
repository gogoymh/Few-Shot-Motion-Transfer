import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os
from torchvision import transforms

from new_architecture33 import Discriminator, Generator
from new_dataset4 import pair_set, joint_set
from ContentLoss import LossCnt


save_path = "/data1/ymh/FSMT/save/new_result70/"
model_name = "/data1/ymh/FSMT/save/" + "generator70.pth"
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

joint_discriminator = Joint_Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

style_discriminator = Style_Discriminator().to(device)
#discriminator = nn.DataParallel(discriminator)
#discriminator().to(device)

param = list(generator.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.0001, betas=(0, 0.9))
optimizer_D_joint = torch.optim.Adam(joint_discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
optimizer_D_style = torch.optim.Adam(style_discriminator.parameters(), lr=0.0001, betas=(0, 0.9))


L1_loss = nn.L1Loss()
Perceptual_loss = LossCnt(VGGFace_body_path='Pytorch_VGGFACE_IR.py', VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)


fake_value = Tensor(batch_size, 1).fill_(1.0).to(device)


#checkpoint = torch.load("/home/compu/ymh/FSMT/save/generator57.pth")
#generator.load_state_dict(checkpoint["g"])
#optimizer_G.load_state_dict(checkpoint["g_optim"])
color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.ToTensor()
        ])

def color_transform(img_batch, device):
    for i in range(img_batch.shape[0]):
        tmp = img_batch[i].cpu()
        tmp = transform(tmp)
        img_batch[i] = tmp.to(device)
    return img_batch

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
    
    ## ---- Joint D ---- ##
    joint_discriminator.train()
    generator.eval()
    
    for i in range(n_critic):
        optimizer_D_joint.zero_grad()
        
        joint, real = pair_loader.__iter__().next()
        query = joint_loader.__iter__().next()
    
        joint = joint.to(device)
        real = real.to(device)
        query = query.to(device)
            
        real = color_transform(real, device)
        
        real_validity = joint_discriminator(real, joint)
        fake_validity = joint_discriminator(real, query)
        
        alpha = Tensor(np.random.random((batch_size, 1, 1, 1)))
        joint_interpolates = (alpha * joint.data + ((1 - alpha) * query.data)).requires_grad_(True)
    
        d_interpolates = joint_discriminator(real, joint_interpolates)

        gradients = autograd.grad(outputs=d_interpolates, inputs=joint_interpolates, grad_outputs=fake_value, create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
        joint_d_loss = -real_validity.mean() + fake_validity.mean() + lambda_gp * gradient_penalty

        joint_d_loss.backward()
        optimizer_D_joint.step()
    
    print("[Joint D:%f]" % joint_d_loss.item(), end=" ")
    
    ## ---- Style D ---- ##
    style_discriminator.train()
    generator.eval()
    
    for i in range(n_critic):
        optimizer_D_style.zero_grad()
        
        joint, real = pair_loader.__iter__().next()
        query = joint_loader.__iter__().next()
    
        joint = joint.to(device)
        real = real.to(device)
        query = query.to(device)
    
        fake = generator(query, real)
        
        real_validity = style_discriminator(real)
        fake_validity = style_discriminator(fake)
        
        alpha = Tensor(np.random.random((batch_size, 1, 1, 1)))
        interpolates = (alpha * real.data + ((1 - alpha) * fake.data)).requires_grad_(True)
    
        d_interpolates = style_discriminator(interpolates)

        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake_value, create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
        style_d_loss = -real_validity.mean() + fake_validity.mean() + lambda_gp * gradient_penalty

        style_d_loss.backward()
        optimizer_D_style.step()
    
    print("[Style D:%f]" % style_d_loss.item(), end=" ")    
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    joint_discriminator.eval()
    style_discriminator.eval()
    
    realistic_fake = generator(joint, real)
    fake = generator(query, real)
    
    joint_fake_validity = joint_discriminator(fake, query)
    style_fake_validity = style_discriminator(fake)
    
    adv_loss = -joint_fake_validity.mean() -style_fake_validity.mean()
    p_loss = Perceptual_loss(fake, real) * 2.5
    l1_loss = L1_loss(realistic_fake, real) * 5
    
    g_loss = adv_loss + p_loss + l1_loss
    g_loss.backward()
    optimizer_G.step()
    print("[G:%f] [P:%f] [L1:%f]" % (g_loss.item(), p_loss.item(), l1_loss.item()))

    if iteration % save == 0:
        for_save = torch.cat((fake.data[:2], real.data[:2]), dim=0)
        
        save_image(for_save, os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd_joint': joint_discriminator.state_dict(),
                    'd_style': style_discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_joint_optim': optimizer_D_joint.state_dict(),
                    'd_style_optim': optimizer_D_style.state_dict()
                    }, model_name)













