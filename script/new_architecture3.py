import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

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

class Style(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU()
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
        self.norm1 = nn.InstanceNorm2d(out_channel)
        self.norm2 = nn.InstanceNorm2d(out_channel)
        
    def forward(self, x):
        res = self.conv3(x)
        res = self.pool(res)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        out = self.pool(out)
        
        out = out + res
        
        return out
    
class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.LeakyReLU()
        
        self.style1 = Feature(out_channel)
        self.style2 = Feature(out_channel)
        self.style3 = Feature(out_channel)
        self.style4 = Feature(out_channel)
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        
    def forward(self, x, y, z):
        
        out = self.upsample(x)
        
        out = self.conv1(out)
        out = self.style1(out, z)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.style2(out, z)
        out = self.relu(out)
        
        out = torch.cat((out, y), dim=1)
        
        out = self.conv3(out)
        out = self.style3(out, z)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.style4(out, z)
        out = self.relu(out)
        
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(3, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)

        self.style1 = Style(3, 32)
        self.style2 = Style(32, 64)
        self.style3 = Style(64, 128)
        self.style4 = Style(128, 256)
        
        self.out1 = nn.utils.spectral_norm(nn.Conv2d(32, 16, 3, 1, 1, bias=True))
        self.out2 = nn.utils.spectral_norm(nn.Conv2d(16, 3, 3, 1, 1, bias=True))
        self.relu = nn.LeakyReLU()
        
        self.tanh = nn.Tanh()
        
        self.init_help = nn.Parameter(torch.randn((1,1,20,12)))
        
    def forward(self, x, y):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) + self.init_help.expand((x.shape[0],512,20,12))
                
        y1 = self.style1(y)
        y2 = self.style2(y1)
        y3 = self.style3(y2)
        y4 = self.style4(y3)
        
        x = self.up1(x5, x4, y4)
        x = self.up2(x, x3, y3)
        x = self.up3(x, x2, y2)
        x = self.up4(x, x1, y1)
        
        x = self.relu(self.out1(x))
        x = self.tanh(self.out2(x))
        
        return x

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
        
        self.block1 = ResBlockDown(6, 128)
        self.block2 = ResBlockDown(128, 256)
        self.block3 = ResBlockDown(256, 512)
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
    b = Generator().to(device)
    c = b(a, a)
    print(c.shape)