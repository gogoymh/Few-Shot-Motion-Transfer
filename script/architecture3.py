import torch
import torch.nn as nn

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
    
    
    adain = std_style * (feature - mean_feat)/std_feat + mean_style
    
    #print(adain.shape)
    
    adain = adain.view(B,C,H,W)
    
    #print(adain.shape)
    #print("-"*20)
    
    return adain

class Style_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Style_ResBlock, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace = False)
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        
        self.mean1 = nn.utils.spectral_norm(nn.Linear(512, in_channel))
        self.std1 = nn.utils.spectral_norm(nn.Linear(512, in_channel))
        self.mean2 = nn.utils.spectral_norm(nn.Linear(512, out_channel))
        self.std2 = nn.utils.spectral_norm(nn.Linear(512, out_channel))
        
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
    def forward(self, x, y):        
        res = x
        res = self.conv3(res)
        
        mean1 = self.mean1(y).unsqueeze(2)
        std1 = self.std1(y).unsqueeze(2)
        
        out = adaIN(x, mean1, std1)
        out = self.relu(out)
        out = self.conv1(out)
        
        mean2 = self.mean2(y).unsqueeze(2)
        std2 = self.std2(y).unsqueeze(2)

        out = adaIN(out, mean2, std2)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + res
        
        return out

class Generator1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = Style_ResBlock(3, 32)
        self.block2 = Style_ResBlock(32, 64)
        self.block3 = Style_ResBlock(64, 128)
        self.block4 = Style_ResBlock(128, 256)
        self.block5 = Style_ResBlock(256, 512)
        
        self.block6 = Style_ResBlock(512, 256)
        self.block7 = Style_ResBlock(256, 128)
        self.block8 = Style_ResBlock(128, 64)
        self.block9 = Style_ResBlock(64, 32)
        
        self.block10 = Style_ResBlock(512, 256)
        self.block11 = Style_ResBlock(256, 128)
        self.block12 = Style_ResBlock(128, 64)
        self.block13 = Style_ResBlock(64, 32)
   
        self.block14 = nn.utils.spectral_norm(nn.Conv2d(32, 3, 3, 1, 1, bias=False))
        
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.tanh = nn.Tanh()
                                             
    def forward(self, x, y): # x: (B, 3, 256, 256), y: (B, 512)
        
        out1 = self.block1(x, y) # (B, 32, 256, 256)
        out2 = self.block2(self.down(out1), y) # (B, 64, 128, 128)
        out3 = self.block3(self.down(out2), y) # (B, 128, 64, 64)
        out4 = self.block4(self.down(out3), y) # (B, 256, 32, 32)
        out5 = self.block5(self.down(out4), y) # (B, 512, 16, 16)
        
        out6 = self.block6(torch.cat((self.block10(self.up(out5), y), out4), dim=1), y) # (B, 256, 32, 32)
        out7 = self.block7(torch.cat((self.block11(self.up(out6), y), out3), dim=1), y) # (B, 128, 64, 64)
        out8 = self.block8(torch.cat((self.block12(self.up(out7), y), out2), dim=1), y) # (B, 64, 128, 128)
        out9 = self.block9(torch.cat((self.block13(self.up(out8), y), out1), dim=1), y) # (B, 32, 256, 256)

        out10 = self.block14(out9)
        
        return self.tanh(out10)

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))

    def forward(self, x):
        res = x

        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
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
        
        self.block1 = ResBlockDown(6, 64)
        self.block2 = ResBlockDown(64, 128)
        self.block3 = ResBlockDown(128, 256)
        self.block4 = ResBlockDown(256, 512)
        self.block5 = ResBlockDown(512, 512)
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
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.pool(out)
        
        out = torch.flatten(out, 1)
        
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace = False)
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        
        self.in1 = nn.InstanceNorm2d(in_channel)
        self.in2 = nn.InstanceNorm2d(out_channel)
        
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
    def forward(self, x):        
        res = x
        res = self.conv3(res)
        
        out = self.in1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.in2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + res
        
        return out

class Generator2(nn.Module):
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
   
        self.block14 = nn.utils.spectral_norm(nn.Conv2d(32, 3, 3, 1, 1, bias=False))
        
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.tanh = nn.Tanh()
                                             
    def forward(self, x): # x: (B, 3, 256, 256)
        
        out1 = self.block1(x) # (B, 32, 256, 256)
        out2 = self.block2(self.down(out1)) # (B, 64, 128, 128)
        out3 = self.block3(self.down(out2)) # (B, 128, 64, 64)
        out4 = self.block4(self.down(out3)) # (B, 256, 32, 32)
        out5 = self.block5(self.down(out4)) # (B, 512, 16, 16)
        
        out6 = self.block6(torch.cat((self.block10(self.up(out5)), out4), dim=1)) # (B, 256, 32, 32)
        out7 = self.block7(torch.cat((self.block11(self.up(out6)), out3), dim=1)) # (B, 128, 64, 64)
        out8 = self.block8(torch.cat((self.block12(self.up(out7)), out2), dim=1)) # (B, 64, 128, 128)
        out9 = self.block9(torch.cat((self.block13(self.up(out8)), out1), dim=1)) # (B, 32, 256, 256)

        out10 = self.block14(out9)
        
        return self.tanh(out10)

if __name__ == "__main__":
    a = torch.randn((1,3,256,256))
    b = torch.randn((1,3,256,256))
    c = Generator2()
    d = Embedder()
    e = Discriminator()
    
    g = c(b)
    print(g.shape)
    
    
    
    