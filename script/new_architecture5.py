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
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        
    def forward(self, x, z):
        
        out = self.upsample(x)

        out = self.style1(out, z)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.style2(out, z)
        out = self.relu(out)
        out = self.conv2(out)
        
        return out
      
class Content(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
                
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU()
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
    def forward(self, x):
        res = self.conv3(x)
        res = self.pool(res)
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        
        out = out + res
        
        return out
    
class Style(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU()
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
    def forward(self, x):
        res = self.conv3(x)
        res = self.pool(res)
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        
        out = out + res
        
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.content = nn.Sequential(
            Content(3,16),
            Content(16,32),
            Content(32,64),
            Content(64,128)
            )
        
        self.style1 = Style(3,16)
        self.style2 = Style(16,32)
        self.style3 = Style(32,64)
        self.style4 = Style(64,128)
        #self.style5 = Style(128,256)
        #self.style6 = Style(256,512)
        
        #self.up1 = UpBlock(512,256)
        #self.up2 = UpBlock(256,128)
        self.up3 = UpBlock(128,64)
        self.up4 = UpBlock(64,32)
        self.up5 = UpBlock(32,16)
        self.up6 = UpBlock(16,3)
        
        self.init_param = nn.Parameter(torch.randn((1,1,5,3)))
        
        self.tanh = nn.Tanh()
        
    def forward(self, landmark, style):
        
        content = self.content(landmark)
        
        style1 = self.style1(style)
        style2 = self.style2(style1)
        style3 = self.style3(style2)
        style4 = self.style4(style3)
        #style5 = self.style5(style4)
        #style6 = self.style6(style5)
        '''
        out = noise + self.init_param.expand((noise.shape[0],1,5,3))
        out = out.expand((-1,512,5,3))
        
        out = self.up1(out, style6)
        out = self.up2(out, style5)
        
        out = out + content
        '''
        out = self.up3(content, style4)
        out = self.up4(out, style3)
        out = self.up5(out, style2)
        out = self.up6(out, style1)
        
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
        print(out.shape)
        out = self.block2(out)
        print(out.shape)
        out = self.block3(out)
        print(out.shape)
        out = self.block4(out)
        print(out.shape)
        out = self.block5(out)
        print(out.shape)
        out = self.block6(out)
        print(out.shape)
        out = self.block7(out)
        print(out.shape)
        out = self.pool(out)
        print(out.shape)
        
        out = torch.flatten(out, 1)
        print(out.shape)
        out = self.relu(self.fc1(out))
        print(out.shape)
        out = self.fc2(out)
        print(out.shape)
        
        return out

if __name__ == "__main__":
    device = torch.device("cuda:0")
    a = torch.randn((1,3,320,192)).to(device)
    #b = torch.randn((1,1,5,3)).to(device)
    #c = Generator().to(device)
    #d = c(a,a,b)
    #print(d.shape)
    e = Discriminator().to(device)
    f = e(a,a)
    #print(f.shape)

