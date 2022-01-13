import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        mid_channel = in_channel // 2
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x):
        
        query = self.conv1(x)
        key = self.conv2(x)
        gate = self.softmax(query * key)
        
        value = self.conv3(x)
        out = gate * value
        out = self.conv4(out)
        out = self.pool(out)
        
        res = self.conv5(x)
        res = self.pool(res)
        
        out = out + res
        
        return out
    
class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Sequential(
            Attention(6,32),
            Attention(32,64),
            Attention(64,128),
            Attention(128,256),
            Attention(256,512),
            Attention(512,1024)         
            )        
        
    def forward(self, x):
        return self.embed(x)

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.LeakyReLU()
        
        self.norm1 = nn.InstanceNorm2d(in_channel)
        self.norm2 = nn.InstanceNorm2d(in_channel)
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size=3, bias=False))
            )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, bias=False))
            )
        
    def forward(self, x):
        
        out = self.upsample(x)

        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.init_param = nn.Parameter(torch.randn((1,1,5,3)))
        
        self.embed = Embedder()
        
        self.generate = nn.Sequential(
            UpBlock(1024,512),
            UpBlock(512,256),
            UpBlock(256,128),
            UpBlock(128,64),
            UpBlock(64,32),
            UpBlock(32,3)
            )
        
        self.tanh = nn.Tanh()
        
    def forward(self, x, y, noise):
        
        x = self.embed(torch.cat((x,y), dim=1))
        
        x = x + self.init_param + noise
        
        x = self.generate(x)
        
        return self.tanh(x)

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
        
        #right
        self.conv_r1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, bias=False))
            )
        self.conv_r2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, bias=False))
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
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x, y):
        out = self.block1(torch.cat((x, y), dim=1))
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
    e = Generator().to(device)
    f = e(a,a, b)
    print(f.shape)
    g = Discriminator().to(device)
    h = g(f,a)
    print(h.shape)
