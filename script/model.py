import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ResBlockDown, SelfAttention, ResBlock, ResBlockD, ResBlockUp, adaIN
import math
import sys
import os

#components
class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace=False)
        
        #in 6*256*256
        self.resDown1 = ResBlockDown(3, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 515*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,1)) #out 512*1*1
        
        self.fc1 = nn.utils.spectral_norm(nn.Linear(512, 512))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(512, 512))
        
    def forward(self, x):
        out = self.resDown1(x) #out 64*128*128
        out = self.resDown2(out) #out 128*64*64
        out = self.resDown3(out) #out 256*32*32
        
        out = self.self_att(out) #out 256*32*32
        
        out = self.resDown4(out) #out 512*16*16
        out = self.resDown5(out) #out 512*8*8
        out = self.resDown6(out) #out 512*4*4
        
        out = self.sum_pooling(out) #out 512*1*1
        
        out = torch.flatten(out, 1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        out = out.view(-1,512,1) #out B*512*1
        return out

class Generator(nn.Module):
    P_LEN = 2*(512*2*5 + 512+256 + 256+128 + 128+64 + 64+32 + 32)
    slice_idx = [0,
                512*4, #res1
                512*4, #res2
                512*4, #res3
                512*4, #res4
                512*4, #res5
                512*2 + 256*2, #resUp1
                256*2 + 128*2, #resUp2
                128*2 + 64*2, #resUp3
                64*2 + 32*2, #resUp4
                32*2] #last adain
    for i in range(1, len(slice_idx)):
        slice_idx[i] = slice_idx[i-1] + slice_idx[i]
    
    def __init__(self):
        super(Generator, self).__init__()
        
        #self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(inplace = False)
        
        #Down
        self.resDown1 = ResBlockDown(3, 64, conv_size=9, padding_size=4) #out 64*128*128
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        
        self.self_att_Down = SelfAttention(256) #out 256*32*32
        
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.in4 = nn.InstanceNorm2d(512, affine=True)
        
        #Res
        #in 512*16*16
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.res4 = ResBlock(512)
        self.res5 = ResBlock(512)
        #out 512*16*16
        
        #Up
        #in 512*16*16
        self.resUp1 = ResBlockUp(512, 256) #out 256*32*32
        self.resUp2 = ResBlockUp(256, 128) #out 128*64*64
        
        self.self_att_Up = SelfAttention(128) #out 128*64*64

        self.resUp3 = ResBlockUp(128, 64) #out 64*128*128
        self.resUp4 = ResBlockUp(64, 32, scale=2, conv_size=3, padding_size=1) #out 3*256*256
        self.conv2d = nn.Conv2d(32, 3, 3, padding = 1)
        
        self.p = nn.Parameter(torch.rand(self.P_LEN,512).normal_(0.0,0.02))
            
    def forward(self, y, e):
        
        p = self.p.unsqueeze(0)
        p = p.expand(e.shape[0],self.P_LEN,512)
        e_psi = torch.bmm(p, e) #B, p_len, 1
        print(e_psi.shape)
        
        #Encoding
        out = self.resDown1(y)
        out = self.in1(out)
        
        out = self.resDown2(out)
        out = self.in2(out)
        
        out = self.resDown3(out)
        out = self.in3(out)
        
        out = self.self_att_Down(out)
        
        out = self.resDown4(out)
        out = self.in4(out)
        
        a = e_psi[:, self.slice_idx[0]:self.slice_idx[1], :]
        print(a.shape)
        #Residual
        out = self.res1(out, e_psi[:, self.slice_idx[0]:self.slice_idx[1], :])
        out = self.res2(out, e_psi[:, self.slice_idx[1]:self.slice_idx[2], :])
        out = self.res3(out, e_psi[:, self.slice_idx[2]:self.slice_idx[3], :])
        out = self.res4(out, e_psi[:, self.slice_idx[3]:self.slice_idx[4], :])
        out = self.res5(out, e_psi[:, self.slice_idx[4]:self.slice_idx[5], :])
        
        
        #Decoding
        out = self.resUp1(out, e_psi[:, self.slice_idx[5]:self.slice_idx[6], :])
        
        out = self.resUp2(out, e_psi[:, self.slice_idx[6]:self.slice_idx[7], :])
        
        out = self.self_att_Up(out)

        out = self.resUp3(out, e_psi[:, self.slice_idx[7]:self.slice_idx[8], :])
        
        out = self.resUp4(out, e_psi[:, self.slice_idx[8]:self.slice_idx[9], :])
        
        out = adaIN(out,
                    e_psi[:,
                          self.slice_idx[9]:(self.slice_idx[10]+self.slice_idx[9])//2,
                          :],
                    e_psi[:,
                          (self.slice_idx[10]+self.slice_idx[9])//2:self.slice_idx[10],
                          :]
                   )
        
        out = self.relu(out)
        
        out = self.conv2d(out)
        
        out = self.tanh(out)
        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.relu = nn.LeakyReLU()
        
        #in 6*256*256
        self.resDown1 = ResBlockDown(6, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.res = ResBlockD(512) #out 512*4*4
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,1)) #out 512*1*1
        
        self.fc1 = nn.utils.spectral_norm(nn.Linear(512, 256))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(256, 1))
    
    def forward(self, x, y):
        out = torch.cat((x,y), dim=1) #out B*6*224*224
        
        out1 = self.resDown1(out)
        
        out2 = self.resDown2(out1)
        
        out3 = self.resDown3(out2)
        
        out = self.self_att(out3)
        
        out4 = self.resDown4(out)
        
        out5 = self.resDown5(out4)
        
        out6 = self.resDown6(out5)
        
        out7 = self.res(out6)
        
        out = self.sum_pooling(out7)
        
        out = self.relu(out)
        
        out = torch.flatten(out, 1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
       
        return out

if __name__ == "__main__":
    a = torch.randn((1,3,256,256)) # joint
    b = torch.randn((1,3,256,256)) # style
    c = Generator()
    d = Embedder()
    e = Discriminator()
    f = d(a)
    print(f.shape)
    g = c(b,f)
    print(g.shape)
    