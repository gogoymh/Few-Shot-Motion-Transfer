import torch


from model import Generator, Discriminator
from contrastive_network import Feature
from link import Converter

if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    generator = Generator(256, 512, 8, channel_multiplier=2).to(device)
    discriminator = Discriminator(256, channel_multiplier=2).to(device)
    g_ema = Generator(256, 512, 8, channel_multiplier=2).to(device)
    
    texture = Feature().to(device)
    pose = Feature().to(device)
    
    converter = Converter().to(device)

    #path1 = "/home/compu/ymh/FSMT/stylegan2-pytorch/550000.pt"
    #ckpt1 = torch.load(path1, map_location=lambda storage, loc: storage)
    #generator.load_state_dict(ckpt1["g"], strict=False)
    #discriminator.load_state_dict(ckpt1["d"])
    #g_ema.load_state_dict(ckpt1["g_ema"], strict=False)
    
    #path2 = 
    #ckpt2 = 
    #texture.load_state_dict(ckpt2["model_state_dict"])
    
    #path3 = 
    #ckpt3 = 
    #texture.load_state_dict(ckpt3["model_state_dict"])
    
    print("Loaded.")
    
    a = torch.rand((1,3,256,256))
    print("a", a.shape)
    b = torch.rand((1,3,256,256))
    print("b", b.shape)
    
    c = texture(a)
    print("c", c.shape)
    d = pose(b)
    print("d", d.shape)
    
    e = converter(torch.cat((c,d), dim=1))
    print("e", e.shape)
    
    f, _ = g_ema([e], truncation=1, truncation_latent=None)
    print("f", f.shape)
    
    g = discriminator(f)
    print("g", g.shape)
    h = discriminator(a)
    print("h", h.shape)