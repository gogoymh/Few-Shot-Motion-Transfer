import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        #print("="*30)
        #print(m.weight.data[0,0,0,0])
        
        shape0 = m.weight.data.shape[0]
        shape1 = m.weight.data.shape[1]
        shape2 = m.weight.data.shape[2]
        shape3 = m.weight.data.shape[3]
        #print(m.weight.data.shape)
        
        m.weight.data = m.weight.data/math.sqrt(shape0*shape1*shape2*shape3)
        
        #print(m.weight.data[0,0,0,0])
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        
    if isinstance(m, nn.Linear):
        
        shape0 = m.weight.data.shape[0]
        shape1 = m.weight.data.shape[1]
        m.weight.data = m.weight.data/math.sqrt(shape0*shape1)
        
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=True
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        
        if activation:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, input):
        if self.activation:
            out = F.linear(input, weight = self.weight * self.scale, bias = self.bias * self.lr_mul)
            out = self.relu(out)

        else:
            out = F.linear(
                input, weight = self.weight * self.scale, bias = self.bias * self.lr_mul
            )

        return out

class Feature(nn.Module):
    def __init__(self, in_channel, eps = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.fc1 = EqualLinear(512, in_channel, activation=False)
        self.fc2 = EqualLinear(512, in_channel, activation=False)
        
    def forward(self, x, y):
        
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
            
        std_feat = (torch.std(x, dim = 2) + self.eps).view(B,C,1)
        mean_feat = torch.mean(x, dim = 2).view(B,C,1)
        
        mean = self.fc1(y).unsqueeze(2)
        std = self.fc2(y).unsqueeze(2)
        
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
        self.style2 = Feature(out_channel)
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size = 3, bias=False))
            )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size = 3, bias=False))
            )
        
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size = 1, bias=False))
        
        for m in self.modules():
            _weights_init(m)
        
    def forward(self, x, z):
        
        out = self.upsample(x)
        res = self.conv3(out)
        
        out = self.relu(out)
        out = self.conv1(out)
        out = self.style1(out, z)
        
        out = self.relu(out)
        out = self.conv2(out)
        out = self.style2(out, z)
                
        out = out + res
        
        return out
      
class RGBBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.style = Feature(3)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, 3, kernel_size = 3, bias=True))
            )

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        
        for m in self.modules():
            _weights_init(m)
        
    def forward(self, x, z, prev_rgb=None):
        
        out = self.relu(x)
        out = self.conv(out)
        out = self.style(out, z)       

        if prev_rgb is not None:
            prev_rgb = self.upsample(prev_rgb)
            
            out = out + prev_rgb
        
        return out


class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        mid_channel = in_channel // 2
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False))
        
        for m in self.modules():
            _weights_init(m)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        query = self.conv1(x)
        key = self.conv2(x)
        gate = self.softmax(query * key)
        
        value = self.conv3(x)
        out = gate * value
        
        out = self.conv4(out)
        
        out = out + x
        
        return out



class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedder = nn.Sequential(
            EqualLinear(512, 512),
            EqualLinear(512, 512),
            EqualLinear(512, 512),
            EqualLinear(512, 512),
            EqualLinear(512, 512),
            EqualLinear(512, 512),
            EqualLinear(512, 512)
            )
        
        self.up1 = UpBlock(512,512)
        self.up2 = UpBlock(512,256)
        self.up3 = UpBlock(256,128)
        self.up4 = UpBlock(128,64)
        self.up5 = UpBlock(64,32)
        self.up6 = UpBlock(32,16)
        
        self.rgb1 = RGBBlock(512)
        self.rgb2 = RGBBlock(256)
        self.rgb3 = RGBBlock(128)
        self.rgb4 = RGBBlock(64)
        self.rgb5 = RGBBlock(32)
        self.rgb6 = RGBBlock(16)
        
        self.init_param = nn.Parameter(torch.randn((1,512,5,3))/math.sqrt(512*5*3))
        
        self.conv = ResBlockD(3)
        
        for m in self.modules():
            _weights_init(m)
        
    def forward(self, style):
        
        out = self.init_param.expand((style.shape[0],512,5,3))
        style = self.embedder(style)
        #print(style)
        
        out = self.up1(out, style)
        rgb = self.rgb1(out, style)
        
        out = self.up2(out, style)
        rgb = self.rgb2(out, style, rgb)
        
        out = self.up3(out, style)
        rgb = self.rgb3(out, style, rgb)
        
        out = self.up4(out, style)
        rgb = self.rgb4(out, style, rgb)
        
        out = self.up5(out, style)
        rgb = self.rgb5(out, style, rgb)
        
        out = self.up6(out, style)
        rgb = self.rgb6(out, style, rgb)
        
        rgb = self.conv(rgb)
        
        return rgb

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.relu_inplace = nn.LeakyReLU(inplace = False)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False))
        
        for m in self.modules():
            _weights_init(m)

    def forward(self, x):
        res = x

        res = self.conv_l1(res)
        res = self.avg_pool2d(res)
        
        out = self.relu(x)
        out = self.conv_r1(out)
        #out = self.norm1(out)
        
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        #out = self.norm2(out)
                
        out = self.avg_pool2d(out)
        
        out = res + out
        
        return out

class ResBlockD(nn.Module):
    def __init__(self, in_channel):
        super(ResBlockD, self).__init__()
        
        #using no ReLU method
        
        #general
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        
        for m in self.modules():
            _weights_init(m)
        
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + res
        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = ResBlockDown(3, 64)
        self.block2 = ResBlockDown(64, 128)
        #self.block2_1 = Attention(128, 128)
        self.block3 = ResBlockDown(128, 256)
        self.block3_1 = Attention(256, 256)
        self.block4 = ResBlockDown(256, 512)
        #self.block4_1 = Attention(512, 512)
        self.block5 = ResBlockDown(512, 512)
        #self.block5_1 = Attention(1024, 1024)
        self.block6 = ResBlockDown(512, 512)
        #self.block6_1 = Attention(2048, 2048)
        self.block7 = ResBlockDown(512, 512)
        self.block8 = ResBlockD(512)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        #self.fc1 = nn.utils.spectral_norm(nn.Linear(2048, 256))
        self.fc2 = nn.Linear(512, 1)
        
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        
        out = self.block1(x)
        out = self.block2(out)
        #out = self.block2_1(out)
        out = self.block3(out)
        #out = self.block3_1(out)
        out = self.block4(out)
        #out = self.block4_1(out)
        out = self.block5(out)
        #out = self.block5_1(out)
        out = self.block6(out)
        #out = self.block6_1(out)
        out = self.block7(out)
        out = self.block8(out)

        out = self.pool(out)
        out=  self.relu(out)
        
        out = torch.flatten(out, 1)
        #out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    L1_loss = nn.L1Loss()
    device = torch.device("cuda:0")
    generator = Generator().to(device)
    #embedder = Embedder().to(device)
    discriminator = Discriminator().to(device)
    
    #joint = torch.randn((1,3,320,192)).to(device)
    
    for i in range(12):
        real = torch.randn((1,512)).to(device)
    
        #style = embedder(real)
        fake = generator(real)
        #print(fake.shape)
    
    
        b = fake.squeeze().detach().cpu()
        
        b[0] = b[0]*0.5 + 0.5
        b[1] = b[1]*0.5 + 0.5
        b[2] = b[2]*0.5 + 0.5
        
        b = b.clamp(0,1)
        
        b = b.numpy().transpose(1,2,0)    
    
        plt.imshow(b)
        plt.show()
        plt.close() 
    
    #fake_validity = discriminator(joint)
    #print(fake_validity.shape)
    

    
    
    
    

