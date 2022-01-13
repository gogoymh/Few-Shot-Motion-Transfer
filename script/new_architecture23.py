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

class Attention_style(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        mid_channel = in_channel // 2
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool2d(2)
        #self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        
        query = self.conv1(x)
        key = self.conv2(x)
        gate = self.softmax(query * key)
        
        value = self.conv3(x)
        out = gate * value
        #out = self.relu(out)
        
        out = self.conv4(out)
        #out = self.relu(out)
        out = self.pool(out)
        
        res = self.conv5(x)
        res = self.pool(res)
        
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
            Content(64,256)
            )
        
        self.style1 = Attention_style(6,32)
        self.style2 = Attention_style(32,64)
        self.style3 = Attention_style(64,128)
        self.style4 = Attention_style(128,256)
        self.style5 = Attention_style(256,256)
        self.style6 = Attention_style(256,512)
        
        self.up1 = UpBlock(512,256)
        self.up2 = UpBlock(256,256)
        self.up3 = UpBlock(256,128)
        self.up4 = UpBlock(128,64)
        self.up5 = UpBlock(64,32)
        self.up6 = UpBlock(32,3)
        
        self.init_param = nn.Parameter(torch.randn((1,1,5,3)))
        
        self.tanh = nn.Tanh()
        
    def forward(self, landmark, style, noise):
        
        content = self.content(landmark)
        
        style1 = self.style1(torch.cat((landmark.expand_as(style), style), dim=1))
        style2 = self.style2(style1)
        style3 = self.style3(style2)
        style4 = self.style4(style3)
        style5 = self.style5(style4)
        style6 = self.style6(style5)
        
        style1 = style1.max(dim=0, keepdim=True)[0]
        style2 = style2.max(dim=0, keepdim=True)[0]
        style3 = style3.max(dim=0, keepdim=True)[0]
        style4 = style4.max(dim=0, keepdim=True)[0]
        style5 = style5.max(dim=0, keepdim=True)[0]
        style6 = style6.max(dim=0, keepdim=True)[0]
        
        out = self.init_param.expand((noise.shape[0],512,5,3))
        
        out = self.up1(out, style6)
        
        out = out + noise.expand_as(out)
        
        out = self.up2(out, style5)
        
        out = out + content
        
        out = self.up3(out, style4)
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
    b = torch.randn((1,1,10,6)).to(device)
    c = Generator().to(device)
    d = c(a,a,b)
    print(d.shape)
    e = Discriminator().to(device)
    f = e(d,a)
    print(f.shape)

