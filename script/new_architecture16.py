import torch
import torch.nn as nn

class Feature(nn.Module):
    def __init__(self, in_channel, eps = 1e-5):
        super().__init__()
        
        self.eps = eps
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x, y):
        
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
            
        std_feat = (torch.std(x, dim = 2) + self.eps).view(B,C,1)
        mean_feat = torch.mean(x, dim = 2).view(B,C,1)
        
        mean = self.avgpool(self.conv1(y)).squeeze(3)
        std = self.avgpool(self.conv2(y)).squeeze(3)
        
        new_feature = std * (x - mean_feat)/std_feat + mean
        new_feature = new_feature.view(B,C,H,W)
        
        return new_feature
           

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.LeakyReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.style1 = Feature(in_channel)
        self.style2 = Feature(in_channel)
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True))
        
    def forward(self, x, z):
        
        out = self.upsample(x)

        out = self.style1(out, z)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.style2(out, z)
        out = self.relu(out)
        out = self.conv2(out)
        
        return out
          
class Encode_block(nn.Module):
    def __init__(self, in_channel, out_channel, content_style=True, downsample=True):
        super().__init__()
        
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        
        if downsample:
            self.pool = nn.AvgPool2d(2)
            self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
            self.downsample = True
        else:
            self.downsample = False
        
        if content_style:
            self.norm1 = nn.BatchNorm2d(out_channel)
            self.norm2 = nn.BatchNorm2d(out_channel)            
        else:
            self.norm1 = nn.InstanceNorm2d(out_channel)
            self.norm2 = nn.InstanceNorm2d(out_channel)      
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        if self.downsample:
            out = self.pool(out)
        
            x = self.conv3(x)
            x = self.pool(x)
        
        out = out + x
        
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.content1 = Encode_block(3,16)
        self.content2 = nn.Sequential(
            Encode_block(16,32),
            Encode_block(32,64),
            Encode_block(64,256)
            )
        
        self.style1 = Encode_block(3,32, content_style=False)
        self.style2 = Encode_block(32,64, content_style=False)
        self.style3 = Encode_block(64,128, content_style=False)
        self.style4 = Encode_block(128,256, content_style=False)
        
        helper0 = []
        for _ in range(15):
            helper0.append(Encode_block(256,256, content_style=False, downsample=False))
        self.helper0 = nn.Sequential(*helper0)
        
        helper1 = []
        for _ in range(12):
            helper1.append(Encode_block(128,128, content_style=False, downsample=False))
        self.helper1 = nn.Sequential(*helper1)
        
        helper2 = []
        for _ in range(9):
            helper2.append(Encode_block(64,64, content_style=False, downsample=False))
        self.helper2 = nn.Sequential(*helper2)
        
        helper3 = []
        for _ in range(6):
            helper3.append(Encode_block(32,32, content_style=False, downsample=False))
        self.helper3 = nn.Sequential(*helper3)
        
        helper4 = []
        for _ in range(3):
            helper4.append(Encode_block(3,3, content_style=False, downsample=False))
        self.helper4 = nn.Sequential(*helper4)
                
        self.up1 = UpBlock(256,128)
        self.up2 = UpBlock(128,64)
        self.up3 = UpBlock(64,32)
        self.up4 = UpBlock(32,3)
        
        self.init_param = nn.Parameter(torch.randn((1,1,20,12)))
        
        self.tanh = nn.Tanh()
        
    def forward(self, landmark, style, noise):
        
        content = self.content1(landmark)
        content = content + noise.expand_as(content)
        content = self.content2(content)
        content = content + self.init_param.expand_as(content)
        
        style1 = self.style1(style)
        style2 = self.style2(style1)
        style3 = self.style3(style2)
        style4 = self.style4(style3)
        
        out = self.helper0(content)
        out = self.up1(out, style4)
        
        out = self.helper1(out)
        out = self.up2(out, style3)
        
        out = self.helper2(out)
        out = self.up3(out, style2)
        
        out = self.helper3(out)
        out = self.up4(out, style1)
        
        out = self.helper4(out)
        
        return self.tanh(out)

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
        
        self.norm1 = nn.InstanceNorm2d(out_channel)
        self.norm2 = nn.InstanceNorm2d(out_channel)        

    def forward(self, x):
        res = x

        res = self.conv_l1(res)
        res = self.avg_pool2d(res)
                
        out = self.conv_r1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv_r2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        out = self.avg_pool2d(out)
        
        out = res + out
        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = ResBlockDown(6, 64)
        self.block2 = ResBlockDown(64, 512)
        self.block3 = ResBlockDown(512, 512)
        self.block4 = ResBlockDown(512, 512)
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

if __name__ == "__main__":
    device = torch.device("cuda:0")
    a = torch.randn((1,3,320,192)).to(device)
    b = torch.randn((1,16,1,1)).to(device)
    c = Generator().to(device)
    d = c(a,a,b)
    print(d.shape)
    e = Discriminator().to(device)
    f = e(d,a)
    print(f.shape)

