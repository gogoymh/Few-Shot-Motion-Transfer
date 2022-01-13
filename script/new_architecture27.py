import torch
import torch.nn as nn

class Feature(nn.Module):
    def __init__(self, in_channel, eps = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size = 3, bias=False))
            )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size = 3, bias=False))
            )
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
        self.relu = nn.LeakyReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.style1 = Feature(in_channel)
        self.style2 = Feature(in_channel)
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size = 3, bias=False))
            )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size = 3, bias=False))
            )
        
    def forward(self, x, z):
        
        out = self.upsample(x)

        out = self.style1(out, z)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.style2(out, z)
        out = self.relu(out)
        out = self.conv2(out)
        
        return out
      
class RGBBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.style = Feature(in_channel)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, 3, kernel_size = 3, bias=True))
            )

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)

    def forward(self, x, z, prev_rgb=None):
        
        out = self.style(x, z)
        out = self.relu(out)
        out = self.conv(out)

        if prev_rgb is not None:
            prev_rgb = self.upsample(prev_rgb)
            
            out = out + prev_rgb

        return out
    
class Content(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
                
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size = 3, bias=False))
            )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size = 3, bias=False))
            )
        
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
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()
        
        self.downsample = downsample
        mid_channel = in_channel // 2
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        
        query = self.conv1(x)
        key = self.conv2(x)
        gate = self.softmax(query * key)
        
        value = self.conv3(x)
        out = gate * value
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.relu(out)
        if self.downsample:
            out = self.pool(out)
        
        res = self.conv5(x)
        if self.downsample:
            res = self.pool(res)
        
        out = out + res
        
        return out

class Style(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size = 3, bias=False))
            )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size = 3, bias=False))
            )
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
        
        self.style0 = Attention_style(6,16, False)
        self.style1 = Style(16,32)
        self.style2 = Style(32,64)
        self.style3 = Style(64,128)
        self.style4 = Style(128,256)
        self.style5 = Style(256,256)
        self.style6 = Style(256,512)
        
        self.up1 = UpBlock(512,256)
        self.up2 = UpBlock(256,256)
        self.up3 = UpBlock(256,128)
        self.up4 = UpBlock(128,64)
        self.up5 = UpBlock(64,32)
        self.up6 = UpBlock(32,16)
        
        self.rgb1 = RGBBlock(256)
        self.rgb2 = RGBBlock(256)
        self.rgb3 = RGBBlock(128)
        self.rgb4 = RGBBlock(64)
        self.rgb5 = RGBBlock(32)
        self.rgb6 = RGBBlock(16)
        
        self.init_param = nn.Parameter(torch.randn((1,1,5,3)))
        
    def concat(self, x, y):
        
        ref = None
        for i in range(x.shape[0]):
            cat = torch.cat((x[i].expand_as(y), y), dim=1)
            cat = cat.unsqueeze(0)

            if ref is None:
                ref = cat
            else:
                ref = torch.cat((ref,cat), dim=0)
        
        return ref
        
    def forward(self, landmark, style):
        
        content = self.content(landmark)
        
        ref = self.concat(landmark, style)
        ref = ref.reshape(-1,6,320,192)
        style0 = self.style0(ref)
        style0 = style0.reshape(-1,style.shape[0],16,320,192)
        style0 = style0.max(dim=1, keepdim=True)[0]
        style0 = style0.squeeze(1)
        
        style1 = self.style1(style0)
        style2 = self.style2(style1)
        style3 = self.style3(style2)
        style4 = self.style4(style3)
        style5 = self.style5(style4)
        style6 = self.style6(style5)
        
        out = self.init_param.expand((landmark.shape[0],512,5,3))
        
        out = self.up1(out, style6)
        rgb = self.rgb1(out, style5)
        
        out = self.up2(out, style5)
        out = out + content
        rgb = self.rgb2(out, style4, rgb)
        
        out = self.up3(out, style4)
        rgb = self.rgb3(out, style3, rgb)
        
        out = self.up4(out, style3)
        rgb = self.rgb4(out, style2, rgb)
        
        out = self.up5(out, style2)
        rgb = self.rgb5(out, style1, rgb)
        
        out = self.up6(out, style1)
        rgb = self.rgb6(out, style0, rgb)
        
        return rgb

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
        #right
        self.conv_r1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size = 3, bias=False))
            )
        self.conv_r2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size = 3, bias=False))
            )
        
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
        
        self.relu = nn.LeakyReLU(inplace=True)
        
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
    a1 = torch.randn((8,3,320,192)).to(device)
    a2 = torch.randn((16,3,320,192)).to(device)
    b = torch.randn((8,1,5,3)).to(device)
    c = Generator().to(device)
    d = c(a1,a2)
    print("="*10)
    print(d.shape)
    e = Discriminator().to(device)
    f = e(d,a1)
    print(f.shape)

