import torch
import torch.nn as nn
import torch.autograd as autograd
from torchvision.utils import save_image
from skimage.io import imread, imsave
import os
from torchvision import transforms
import numpy as np
import copy

from new_architecture32 import Generator#, Discriminator

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

#Tensor = torch.FloatTensor
#device = torch.device("cpu")

generator = Generator().to(device)
#discriminator = Discriminator().to(device)
param = list(generator.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.0001, betas=(0, 0.9))
#optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

#path = "/home/compu/ymh/FSMT/save/generator67.pth"
path = "/data1/ymh/FSMT/save/generator70.pth"
checkpoint = torch.load(path)

generator.load_state_dict(checkpoint["g"])
#discriminator.load_state_dict(checkpoint["d"])
optimizer_G.load_state_dict(checkpoint["g_optim"])
#optimizer_D.load_state_dict(checkpoint["d_optim"])

L1_loss = nn.L1Loss()
#L1_loss = VGGPerceptualLoss().to(device)

fake_value = Tensor(1, 1).fill_(1.0).to(device)

change = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]) 

change2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((320,192)),
                transforms.ToTensor()
                ]) 

'''
input_path1 = "/home/compu/ymh/FSMT/dataset/yerin_rendered/"
input_path2 = "/home/compu/ymh/FSMT/dataset/"

save_path = "/home/compu/ymh/FSMT/save/inference67/"
'''

input_path1 = "/data1/ymh/FSMT/dataset/yerin_rendered/"
input_path2 = "/data1/ymh/FSMT/dataset/"

save_path = "/data1/ymh/FSMT/save/inference70/"

reference = imread(os.path.join(input_path2, "monalisa.jpg"))
reference = change(reference)
reference = reference.to(device)
reference = reference.unsqueeze(0)

joint = imread(os.path.join(input_path2, "monalisa_rendered_rendered.png"))
joint = change(joint)
joint = joint.to(device)
joint = joint.unsqueeze(0)

semantic = imread(os.path.join(input_path2, "monalisa_semantic2.png"))
semantic = change2(semantic)
semantic = semantic.to(device)
semantic = semantic.unsqueeze(0)

mask_num = (semantic==1).sum()

'''
for i in range(20):
    print("[Meta test:%d]" % i, end=" ")
    ## ---- D ---- ##
    optimizer_D.zero_grad()
    discriminator.train()
    generator.eval()
        
    fake = generator(joint, reference)
            
    real_validity = discriminator(reference, joint)
    fake_validity = discriminator(fake.detach(), joint)
            
    alpha = Tensor(np.random.random((1, 1, 1, 1)))
    interpolates = (alpha * reference.data + ((1 - alpha) * fake.data)).requires_grad_(True)
        
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
    
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

    d_loss.backward()
    optimizer_D.step()
    
    print("[D loss: %f] [Penelty: %f]" % (d_loss.item(), gradient_penalty.item()), end=" ")
    
    ## ---- Unrolling ---- ##
    backup = copy.deepcopy(discriminator.state_dict())
    backup2 = copy.deepcopy(optimizer_D.state_dict())
    
    for i in range(10):
        optimizer_D.zero_grad()
        discriminator.train()
        generator.eval()

        real_validity = discriminator(reference, joint)
        fake_validity = discriminator(fake.detach(), joint)

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)# + lambda_gp * gradient_penalty
            
        d_loss.backward(create_graph=True)
        optimizer_D.step()
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    discriminator.eval()
    
    fake = generator(joint, reference)
    
    l1_loss = (200 * torch.abs((fake * semantic - reference * semantic))).reshape(1,-1)
    l1_loss = l1_loss.sum(dim=1, keepdim=True)
    l1_loss = l1_loss/mask_num
    l1_loss = l1_loss.mean()
    
    global_l1_loss = L1_loss(fake, reference) * 100
    
    fake_validity = discriminator(fake, joint)
    g_loss = -torch.mean(fake_validity) + l1_loss + global_l1_loss
    g_loss.backward()
    optimizer_G.step()
    print("[L1 loss:%f] [G L1 loss:%f]" % (l1_loss.item(), global_l1_loss.item()))
    
    discriminator.load_state_dict(backup)    
    del backup
        
    optimizer_D.load_state_dict(backup2)
    del backup2
'''

generator.eval()
for index in range(5940):
    print(index)
    joint = imread(os.path.join(input_path1, "frame_%07d_rendered.png" % index))
    #reference = reference_from.__getitem__(0)
    
    joint = change(joint)
    joint = joint.to(device)    
    joint = joint.unsqueeze(0)
    
    #noise = Tensor(np.random.normal(0,1,(1,1,10,6)))
    
    fake = generator(joint, reference)#, noise)
    
    save_image(fake.data[:1], os.path.join(save_path, "frame_%07d.png" % index), nrow=1, normalize=True, range=(-1, 1))
        
    