import torch
import torch.nn as nn

class Feature(nn.Module):
    def __init__(self, in_channel, eps = 1e-6):
        super().__init__()
        
        self.eps = eps
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=0, bias=False))
            )
            
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=0, bias=False))
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
            nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=0, bias=False))
            )
            
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0, bias=False))
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
          
class Encode_block(nn.Module):
    def __init__(self, in_channel, out_channel, content_style=True, downsample=True, residual=True):
        super().__init__()
        
        self.residual = residual
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0, bias=False))
            )
            
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=0, bias=False))
            )
        
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
        
        if self.downsample:
            out = self.pool(out)
        
            x = self.conv3(x)
            x = self.pool(x)
        
        if self.residual:
            out = out + x
        
        out = self.relu(out)
        
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
            )
        
        self.content1 = Encode_block(16,32, residual=False)
        self.content2 = Encode_block(32,64, residual=False)
        self.content3 = Encode_block(64,128, residual=False)
        self.content4 = Encode_block(128,512, residual=False)
        
        self.style1 = Encode_block(3,64, content_style=False)
        self.style2 = Encode_block(64,128, content_style=False)
        self.style3 = Encode_block(128,256, content_style=False)
        self.style4 = Encode_block(256,512, content_style=False)
                
        self.up1 = UpBlock(512,128)
        self.up2 = UpBlock(256,64)
        self.up3 = UpBlock(128,32)
        self.up4 = UpBlock(64,3)
        
        self.attention1 = Attention(256)
        self.attention2 = Attention(128)
        self.attention3 = Attention(64)
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        
        self.tanh = nn.Tanh()
        
    def forward(self, landmark, style, noise):
        
        content = self.stem(landmark)
        
        content1 = self.content1(content)
        content2 = self.content2(content1)
        content3 = self.content3(content2)
        content4 = self.content4(content3)
        
        style1 = self.style1(style)
        style2 = self.style2(style1)
        style3 = self.style3(style2)
        style4 = self.style4(style3)
        
        out = self.conv(noise)
        out = out + content4
        out = self.up1(out, style4)
        
        out = torch.cat((out, content3), dim=1)
        out = self.attention1(out)
        out = self.up2(out, style3)
        
        out = torch.cat((out, content2), dim=1)
        out = self.attention2(out)
        out = self.up3(out, style2)
        
        out = torch.cat((out, content1), dim=1)
        out = self.attention3(out)
        out = self.up4(out, style1)
        
        return self.tanh(out)

class Attention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        
        mid_channel = in_channel // 8
        
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(mid_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        
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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(6, 128, kernel_size=7, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True)
            )
        
        self.block1 = Encode_block(128, 512, content_style=False)
        self.block2 = Encode_block(512, 512, content_style=False)
        self.block3 = Encode_block(512, 512, content_style=False)
        self.attention1 = Attention(512)
        self.block4 = Encode_block(512, 512, content_style=False)
        self.attention2 = Attention(512)
        self.block5 = Encode_block(512, 512, content_style=False)
        self.attention3 = Attention(512)
        self.block6 = Encode_block(512, 512, content_style=False)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1 = nn.utils.spectral_norm(nn.Linear(512, 256, bias=False))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(256, 1, bias=False))
        
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x, y):
        out = self.stem(torch.cat((x,y), dim=1))
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.attention1(out)
        out = self.block4(out)
        out = self.attention2(out)
        out = self.block5(out)
        out = self.attention3(out)
        out = self.block6(out)
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
    #print(d.shape)
    e = Discriminator().to(device)
    f = e(a,a)

