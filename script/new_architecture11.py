import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

EPS = 1e-8

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
    def __init__(self, latent_dim, input_channel, upsample):
        super().__init__()
        
        self.to_style = nn.utils.spectral_norm(nn.Linear(latent_dim, input_channel))

        self.conv = Conv2DMod(input_channel, 3, 1, demod=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        if upsample:
            self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.sample = None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)
        x = x + self.bias

        if prev_rgb is not None:
            if self.sample is not None:
                prev_rgb = self.sample(prev_rgb)
            
            x = x + prev_rgb

        return x

class UpBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True):
        super().__init__()
        
        if upsample:
            self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.sample = None

        self.to_style1 = nn.utils.spectral_norm(nn.Linear(latent_dim, input_channels))
        self.to_noise1 = nn.utils.spectral_norm(nn.Linear(1, filters))
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.utils.spectral_norm(nn.Linear(latent_dim, filters))
        self.to_noise2 = nn.utils.spectral_norm(nn.Linear(1, filters))
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample)

    def forward(self, x, prev_rgb, style, noise):
        if self.sample is not None:
            x = self.sample(x)
        
        noise = noise.unsqueeze(1)
        noise1 = self.to_noise1(noise).unsqueeze(2).unsqueeze(3)
        noise2 = self.to_noise2(noise).unsqueeze(2).unsqueeze(3)

        style1 = self.to_style1(style)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(style)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
                
        rgb = self.to_rgb(x, prev_rgb, style)
        
        return x, rgb

class DownBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters):
        super().__init__()
        
        self.sample = nn.MaxPool2d(2)

        self.to_style1 = nn.utils.spectral_norm(nn.Linear(latent_dim, input_channels))
        self.to_noise1 = nn.utils.spectral_norm(nn.Linear(1, filters))
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.utils.spectral_norm(nn.Linear(latent_dim, filters))
        self.to_noise2 = nn.utils.spectral_norm(nn.Linear(1, filters))
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU()

    def forward(self, x, style, noise):
        
        noise = noise.unsqueeze(1)
        noise1 = self.to_noise1(noise).unsqueeze(2).unsqueeze(3)
        noise2 = self.to_noise2(noise).unsqueeze(2).unsqueeze(3)

        style1 = self.to_style1(style)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(style)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        
        x = self.sample(x)
        
        return x

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.relu_inplace = nn.LeakyReLU(inplace = False)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1,))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

    def forward(self, x):
        res = x
        
        #left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        
        #conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        
        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x) #BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x) #BxC'xHxW
        h_projection = self.conv_h(x) #BxCxHxW
        
        f_projection = torch.transpose(f_projection.view(B,-1,H*W), 1, 2) #BxNxC', N=H*W
        g_projection = g_projection.view(B,-1,H*W) #BxC'xN
        h_projection = h_projection.view(B,-1,H*W) #BxCxN
        
        attention_map = torch.bmm(f_projection, g_projection) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1
        
        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map) #BxCxN
        out = out.view(B,C,H,W)
        
        out = self.gamma*out + x
        return out

class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace=False)
        
        self.resDown1 = ResBlockDown(3, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 515*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.sum_pooling = nn.AdaptiveMaxPool2d((1,1)) #out 512*1*1

    def forward(self, x):
        out = self.resDown1(x) #out 64*128*128
        out = self.resDown2(out) #out 128*64*64
        out = self.resDown3(out) #out 256*32*32
        
        out = self.self_att(out) #out 256*32*32
        
        out = self.resDown4(out) #out 512*16*16
        out = self.resDown5(out) #out 512*8*8
        out = self.resDown6(out) #out 512*4*4
        
        out = self.sum_pooling(out) #out 512*1*1
        out = self.relu(out) #out 512*1*1
        out = out.view(-1,512,1) #out B*512*1
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, capacity=1):
        super().__init__()
        
        self.embed = Embedder()
        
        channels = [32*capacity, 64*capacity, 128*capacity, 256*capacity, 512*capacity]
        
        self.init = DownBlock(latent_dim,3,channels[0])
        self.DownBlocks = nn.ModuleList()
        self.DownBlocks.append(DownBlock(latent_dim,channels[0],channels[1]))
        self.DownBlocks.append(DownBlock(latent_dim,channels[1],channels[2]))
        self.DownBlocks.append(DownBlock(latent_dim,channels[2],channels[3]))
        self.DownBlocks.append(DownBlock(latent_dim,channels[3],channels[4]))
        
        self.mid = UpBlock(latent_dim,channels[4],channels[4],False)
        self.KeepBlocks = nn.ModuleList()
        self.KeepBlocks.append(UpBlock(latent_dim,channels[4],channels[4],False))
        self.KeepBlocks.append(UpBlock(latent_dim,channels[4],channels[4],False))
        
        self.UpBlocks = nn.ModuleList()
        self.UpBlocks.append(UpBlock(latent_dim,channels[4],channels[3]))
        self.UpBlocks.append(UpBlock(latent_dim,channels[3],channels[2]))
        self.UpBlocks.append(UpBlock(latent_dim,channels[2],channels[1]))
        self.UpBlocks.append(UpBlock(latent_dim,channels[1],channels[0]))
        self.UpBlocks.append(UpBlock(latent_dim,channels[0],3))
        
        self.help = nn.Parameter(torch.zeros(1, 512, 10, 6))
        
    def forward(self, joint, style, noise):
        
        style = self.embed(style).squeeze()
        
        out = self.init(joint, style, noise[:,0])
        idx = 1
        for block in self.DownBlocks:
            out = block(out, style, noise[:,idx])
            idx += 1
        
        out = out + self.help
        
        out, rgb = self.mid(out, None, style, noise[:,idx])
        for block in self.KeepBlocks:
            idx += 1
            out, rgb = block(out, rgb, style, noise[:,idx])
        
        for block in self.UpBlocks:
            idx += 1
            out, rgb = block(out, rgb, style, noise[:,idx])
        
        return rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.utils.spectral_norm(nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1)))

        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels, filters, 3, padding=1)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(filters, filters, 3, padding=1)),
            nn.LeakyReLU()
        )

        self.downsample = nn.utils.spectral_norm(nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Discriminator(nn.Module):
    def __init__(self, fmap_max = 512):
        super().__init__()
        num_layers = int(math.log(256, 2))
        num_init_filters = 6

        blocks = []
        filters = [num_init_filters] + [(64) * (2 ** i) for i in range(num_layers+1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        latent_dim = 1024

        self.final_conv = nn.utils.spectral_norm(nn.Conv2d(512, 512, 3, padding=1))
        self.to_logit = nn.utils.spectral_norm(nn.Linear(latent_dim, 1))

    def forward(self, x, y):
        
        x = torch.cat((x,y), dim=1)
        
        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)
        x = torch.flatten(x, 1)
        x = self.to_logit(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0")
    a = torch.randn((2,3,320,192)).to(device)
    h = torch.randn((2,3,320,192)).to(device)
    b = torch.randn((2,15)).to(device)
    d = Generator(512).to(device)
    e = d(a,h,b)
    f = Discriminator().to(device)
    g = f(a,e)