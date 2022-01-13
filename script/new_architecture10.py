import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nodownsample=False):
        super(ResBlock, self).__init__()
        
        self.relu = nn.LeakyReLU()
        
        self.downsample = None
        stride = 1
        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)),
                nn.AvgPool2d(2)
                )
            stride = 2
        
        if nodownsample:
            self.downsample = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False))
            stride = 1
        
        if out_channel < 4:
            mid_channel = out_channel
        else:
            mid_channel = out_channel //4
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, mid_channel, 1, 1, 0, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(mid_channel, mid_channel, 3, stride, 1, bias=False))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(mid_channel, out_channel, 1, 1, 0, bias=False))
        
        self.norm1 = nn.InstanceNorm2d(in_channel, affine=True)
        self.norm2 = nn.InstanceNorm2d(mid_channel, affine=True)
        self.norm3 = nn.InstanceNorm2d(mid_channel, affine=True)

    def forward(self, x):
        #print(x.shape)
        
        out = self.norm1(x)
        out = self.relu(out) 
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        out = x + out
        #print(out.shape)
        return out

class ResNet(nn.Module):
    def __init__(self, init, mid):
        super().__init__()
        
        self.init = nn.utils.spectral_norm(nn.Conv2d(init, 64, 7, 2, 3, bias=False))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(64)
        
        self.block1 = self.make_layer(3, 64, 256)
        self.block2 = self.make_layer(4, 256, 512)
        self.block3 = self.make_layer(mid, 512, 1024)
        self.block4 = self.make_layer(3, 1024, 2048)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.LeakyReLU()
        
    def make_layer(self, num, in_channel, out_channel):
        
        if in_channel == 64:
            layers = [ResBlock(in_channel, out_channel, True)]
        else:
            layers = [ResBlock(in_channel, out_channel)]
        
        for i in range(num-1):
            layers.append(ResBlock(out_channel, out_channel))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        out = self.init(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool(out)
        
        out = torch.flatten(out, 1)
        
        return out

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)
        
        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, sample):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        if sample == "upsample":
            self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif sample == "downsample":
            self.sample = nn.MaxPool2d(2)       
        else:
            raise Exception('wrong sampling.')
            
        self.bias = nn.Parameter(torch.randn(1, 3, 1, 1))

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)
        x = x + self.bias

        if prev_rgb is not None:
            prev_rgb = self.sample(prev_rgb)
            #print("*"*3, x.shape, prev_rgb.shape)
            x = x + prev_rgb

        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, sample):
        super().__init__()
        
        self.upsample = False
        self.downsample = False
        
        if sample == "upsample":
            self.upsample = True
            self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            sample_rgb = "upsample"
            
        elif sample == "downsample":
            self.downsample = True
            self.sample = nn.MaxPool2d(2)
            sample_rgb = "downsample"
            
        else:
            raise Exception('wrong sampling.')

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU()
        self.to_rgb = RGBBlock(latent_dim, filters, sample_rgb)

    def forward(self, x, prev_rgb, style, noise):
        #print("="*10)
        #print(x.shape)
        if self.upsample:
            x = self.sample(x)
        
        noise = noise.unsqueeze(1)
        noise1 = self.to_noise1(noise).unsqueeze(2).unsqueeze(3)
        noise2 = self.to_noise2(noise).unsqueeze(2).unsqueeze(3)

        style1 = self.to_style1(style)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1.expand_as(x))

        style2 = self.to_style2(style)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2.expand_as(x))
        
        if self.downsample:
            x = self.sample(x)
        #print(x.shape)
        
        rgb = self.to_rgb(x, prev_rgb, style)
        
        return x, rgb

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.style = ResNet(3,6)
        
        self.init = GeneratorBlock(2048,3,32,"downsample")
        
        self.generator_blocks = nn.ModuleList()
        
        self.generator_blocks.append(GeneratorBlock(2048,32,64,"downsample"))
        self.generator_blocks.append(GeneratorBlock(2048,64,128,"downsample"))
        self.generator_blocks.append(GeneratorBlock(2048,128,256,"downsample"))
        self.generator_blocks.append(GeneratorBlock(2048,256,512,"downsample"))
        self.generator_blocks.append(GeneratorBlock(2048,512,256,"upsample"))
        self.generator_blocks.append(GeneratorBlock(2048,256,128,"upsample"))
        self.generator_blocks.append(GeneratorBlock(2048,128,64,"upsample"))
        self.generator_blocks.append(GeneratorBlock(2048,64,32,"upsample"))
        self.generator_blocks.append(GeneratorBlock(2048,32,3,"upsample"))
        
        self.help = nn.Parameter(torch.randn(1, 512, 10, 6))
        
    def forward(self, joint, style, noise):
        
        style = self.style(style)
        
        out, rgb = self.init(joint, None, style, noise[:,0])
        idx = 1
        for block in self.generator_blocks:
            out, rgb = block(out, rgb, style, noise[:,idx])
            
            if idx == 4:
                out = out + self.help.expand_as(out)
                
            idx += 1
        
        return rgb

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = ResNet(6,23)
        self.fc = nn.Linear(2048,1)
        
    def forward(self, x, y):
        return self.fc(self.encoder(torch.cat((x,y), dim=1)))

if __name__ == "__main__":
    device = torch.device("cuda:0")
    a = torch.randn((2,3,320,192)).to(device)
    b = torch.randn((2,10)).to(device)
    c = Generator().to(device)
    d = c(a,a,b)
    print(d.shape)
    e = Discriminator().to(device)
    f = e(a,d)
    print(f.shape)