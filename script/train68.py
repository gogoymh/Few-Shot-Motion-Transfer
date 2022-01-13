import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import copy
import numpy as np
from torchvision.utils import save_image
import os

from new_architecture31 import Generator, Discriminator, Embedder
from new_dataset3 import pair_set
from ContentLoss import LossCnt, LossAdv, LossDSCreal, LossDSCfake


save_path = "/data1/ymh/FSMT/save/new_result68/"
model_name = "/data1/ymh/FSMT/save/" + "generator68.pth"
'''
save_path = "/home/compu/ymh/FSMT/save/new_result62/"
model_name = "/home/compu/ymh/FSMT/save/" + "generator62.pth"
'''
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

total = 800000
batch_size = 8
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

embedder = Embedder().to(device)

param = list(generator.parameters()) + list(embedder.parameters())
optimizer_G = torch.optim.Adam(param, lr=0.0001, betas=(0, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))


L1_loss = LossCnt(VGGFace_body_path='Pytorch_VGGFACE_IR.py', VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
Adv_loss = LossAdv()
DSC_real = LossDSCreal()
DSC_fake = LossDSCfake()


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
    embedder.train()
    
    joint, real = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)
    
    style = embedder(real)
    fake = generator(joint, style)

    l1_loss = L1_loss(fake, real)
    
    l1_loss.backward()
    optimizer_G.step()
    print("[Pre iteration:%d] [L1:%f]" % (pre, l1_loss.item()))

for iteration in range(total):
    print("[Iteration:%d]" % iteration, end=" ")
    joint, real = pair_loader.__iter__().next()
    
    joint = joint.to(device)
    real = real.to(device)

    ## ---- D ---- ##
    discriminator.train()
    generator.eval()
    embedder.eval()
    optimizer_D.zero_grad()
    
    with torch.no_grad():
        style = embedder(real)
        fake = generator(joint, style)
    
    real_validity, _ = discriminator(real, joint, style)
    fake_validity, _ = discriminator(fake, joint, style)
        
    d_loss = DSC_real(real_validity) + DSC_fake(fake_validity)

    d_loss.backward()
    optimizer_D.step()
    
    print("[D:%f]" % d_loss.item(), end=" ")
    
    ## ---- G ---- ##
    optimizer_G.zero_grad()
    generator.train()
    embedder.train()
    discriminator.eval()
    
    style = embedder(real)
    fake = generator(joint, style)
    
    _, real_feature = discriminator(real, joint, style)
    fake_validity, fake_feature = discriminator(fake, joint, style)
    
    adv_loss = Adv_loss(fake_validity, fake_feature, real_feature)
    l1_loss = L1_loss(fake, real)
    
    g_loss = adv_loss + l1_loss
    g_loss.backward()
    optimizer_G.step()
    print("[G:%f] [L1:%f]" % (g_loss.item(), l1_loss.item()))
    
    if iteration % save == 0:
        save_image(fake.data[:2], os.path.join(save_path, "%06d_image.png" % iteration), nrow=2, normalize=True, range=(-1, 1))
        
        torch.save({'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'g_optim': optimizer_G.state_dict(),
                    'd_optim': optimizer_D.state_dict()
                    }, model_name)













