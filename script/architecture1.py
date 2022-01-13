import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

def adaIN(feature, mean_style, std_style, eps = 1e-5):
    B,C,H,W = feature.shape
    
    #print("-"*20)
    #print(feature.shape)
    
    #print(mean_style.shape)
    #print(std_style.shape)
    
    feature = feature.view(B,C,-1)
            
    std_feat = (torch.std(feature, dim = 2) + eps).view(B,C,1)
    mean_feat = torch.mean(feature, dim = 2).view(B,C,1)
    
    #print(std_feat.shape)
    #print(mean_feat.shape)
    #print("-"*20)
    
    adain = std_style * (feature - mean_feat)/std_feat + mean_style
    
    adain = adain.view(B,C,H,W)
    return adain

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        
        #using no ReLU method
        #self.noise1 = NoiseInjection()
        #self.noise2 = NoiseInjection()
        
        #general
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        
        #right
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
    def forward(self, x, mean1, std1, mean2, std2):        
        res = x
        res = self.conv3(res)
        
        #out = self.noise1(x)
        out = adaIN(x, mean1, std1)
        out = self.relu(out)
        out = self.conv1(out)
        #out = self.noise2(out)
        out = adaIN(out, mean2, std2)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + res
        
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = ResBlock(3, 32)
        self.block2 = ResBlock(32, 64)
        self.block3 = ResBlock(64, 128)
        self.block4 = ResBlock(128, 256)
        self.block5 = ResBlock(256, 512)
        
        self.block6 = ResBlock(512, 256)
        self.block7 = ResBlock(256, 128)
        self.block8 = ResBlock(128, 64)
        self.block9 = ResBlock(64, 32)
        
        self.block10 = ResBlock(512, 256)
        self.block11 = ResBlock(256, 128)
        self.block12 = ResBlock(128, 64)
        self.block13 = ResBlock(64, 32)
        #self.block14 = ResBlock(32, 3)        
        self.block14 = nn.utils.spectral_norm(nn.Conv2d(32, 3, 3, 1, 1, bias=False))
        
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.tanh = nn.Tanh()
                                             
    def forward(self, x, e): # x: (B, 3, 256, 256), e: (B, 56)
        
        e = e.unsqueeze(2).unsqueeze(3) # (B, 56, 1, 1)
        
        out1 = self.block1(x, e[:,0,:,:], e[:,1,:,:], e[:,2,:,:], e[:,3,:,:]) # (B, 32, 256, 256)
        out2 = self.block2(self.down(out1), e[:,4,:,:], e[:,5,:,:], e[:,6,:,:], e[:,7,:,:]) # (B, 64, 128, 128)
        out3 = self.block3(self.down(out2), e[:,8,:,:], e[:,9,:,:], e[:,10,:,:], e[:,11,:,:]) # (B, 128, 64, 64)
        out4 = self.block4(self.down(out3), e[:,12,:,:], e[:,13,:,:], e[:,14,:,:], e[:,15,:,:]) # (B, 256, 32, 32)
        out5 = self.block5(self.down(out4), e[:,16,:,:], e[:,17,:,:], e[:,18,:,:], e[:,19,:,:]) # (B, 512, 16, 16)
        
        out6 = self.block6(torch.cat((self.block10(self.up(out5), e[:,20,:,:], e[:,21,:,:], e[:,22,:,:], e[:,23,:,:]), out4), dim=1), e[:,24,:,:], e[:,25,:,:], e[:,26,:,:], e[:,27,:,:]) # (B, 256, 32, 32)
        out7 = self.block7(torch.cat((self.block11(self.up(out6), e[:,28,:,:], e[:,29,:,:], e[:,30,:,:], e[:,31,:,:]), out3), dim=1), e[:,32,:,:], e[:,33,:,:], e[:,34,:,:], e[:,35,:,:]) # (B, 128, 64, 64)
        out8 = self.block8(torch.cat((self.block12(self.up(out7), e[:,36,:,:], e[:,37,:,:], e[:,38,:,:], e[:,39,:,:]), out2), dim=1), e[:,40,:,:], e[:,41,:,:], e[:,42,:,:], e[:,43,:,:]) # (B, 64, 128, 128)
        out9 = self.block9(torch.cat((self.block13(self.up(out8), e[:,44,:,:], e[:,45,:,:], e[:,46,:,:], e[:,47,:,:]), out1), dim=1), e[:,48,:,:], e[:,49,:,:], e[:,50,:,:], e[:,51,:,:]) # (B, 32, 256, 256)
        #out10 = self.block14(out9, e[:,52,:,:], e[:,53,:,:], e[:,54,:,:], e[:,55,:,:])
        out10 = self.block14(out9)
        
        return self.tanh(out10)

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))

    def forward(self, x):
        res = x
        
        #left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = ResBlockDown(6, 32)
        self.block2 = ResBlockDown(32, 64)
        self.block3 = ResBlockDown(64, 128)
        self.block4 = ResBlockDown(128, 256)
        self.block5 = ResBlockDown(256, 512)
        self.block6 = ResBlockDown(512, 512)
        self.block7 = ResBlockDown(512, 512)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1 = nn.utils.spectral_norm(nn.Linear(512, 256))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(256, 1))
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x, y):
        
        out = self.block1(torch.cat((x,y), dim=1))
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.pool(out)
        
        out = torch.flatten(out, 1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = ResBlockDown(3, 32)
        self.block2 = ResBlockDown(32, 64)
        self.block3 = ResBlockDown(64, 128)
        self.block4 = ResBlockDown(128, 256)
        self.block5 = ResBlockDown(256, 512)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1 = nn.utils.spectral_norm(nn.Linear(512, 256))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(256, 52))
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.pool(out)
        
        out = torch.flatten(out, 1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    a = torch.randn((1,3,256,256))
    b = torch.randn((1,3,256,256))
    c = Generator()
    d = Embedder()
    e = Discriminator()
    f = d(a)
    g = c(b, f)
    h = e(g, b)
    