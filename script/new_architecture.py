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
        
        self.content = Feature(in_channel)
        self.style = Feature(in_channel)
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        
    def forward(self, x, y, z):
        
        out = self.upsample(x)

        out = self.content(out, y)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.style(out, z)
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
        
        self.content128 = Content(3,16)
        self.content64 = Content(16,32)
        self.content32 = Content(32,64)
        self.content16 = Content(64,128)
        self.content8 = Content(128,256)
        self.content4 = Content(256,512)
        
        self.style128 = Style(3,16)
        self.style64 = Style(16,32)
        self.style32 = Style(32,64)
        self.style16 = Style(64,128)
        self.style8 = Style(128,256)
        self.style4 = Style(256,512)
        
        self.up8 = UpBlock(512,256)
        self.up16 = UpBlock(256,128)
        self.up32 = UpBlock(128,64)
        self.up64 = UpBlock(64,32)
        self.up128 = UpBlock(32,16)
        self.up256 = UpBlock(16,3)
        
        self.init_param = nn.Parameter(torch.randn((1,1,5,3)))
        
        self.tanh = nn.Tanh()
        
    def forward(self, landmark, style, noise):
        
        content1 = self.content128(landmark)
        content2 = self.content64(content1)
        content3 = self.content32(content2)
        content4 = self.content16(content3)
        content5 = self.content8(content4)
        content6 = self.content4(content5)
        
        style1 = self.style128(style)
        style2 = self.style64(style1)
        style3 = self.style32(style2)
        style4 = self.style16(style3)
        style5 = self.style8(style4)
        style6 = self.style4(style5)
                
        out = noise + self.init_param.expand((noise.shape[0],1,5,3))
        out = out.expand((-1,512,5,3))
                
        out = self.up8(out, content6, style6)
        out = self.up16(out, content5, style5)
        out = self.up32(out, content4, style4)
        out = self.up64(out, content3, style3)
        out = self.up128(out, content2, style2)
        out = self.up256(out, content1, style1)
                
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
        
        self.block1 = ResBlockDown(3, 128)
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
        
    def forward(self, x):
        
        out = self.block1(x)
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
    b = torch.randn((1,1,5,3)).to(device)
    c = Generator().to(device)
    d = c(a,a,b)
    print(d.shape)

