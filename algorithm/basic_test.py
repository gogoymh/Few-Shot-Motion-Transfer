import torch
from torchvision import utils


from model import Generator#, Discriminator





if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    #generator = Generator(256, 512, 8, channel_multiplier=2).to(device)
    #discriminator = Discriminator(256, channel_multiplier=2).to(device)
    g_ema = Generator(256, 512, 8, channel_multiplier=2).to(device)


    #path = "C://유민형//개인 연구//Few-Shot Motion Transfer//algorithm//550000.pt"
    path = "/home/compu/ymh/FSMT/stylegan2-pytorch/550000.pt"
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    
    #generator.load_state_dict(ckpt["g"], strict=False)
    #discriminator.load_state_dict(ckpt["d"])
    g_ema.load_state_dict(ckpt["g_ema"], strict=False)
    print("Loaded.")
    
    for i in range(5):
        sample_z = torch.randn(1, 512, device=device)

        sample, _ = g_ema([sample_z], truncation=1, truncation_latent=None)
           
        utils.save_image(
            sample,
            f'sample/{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
            )
        
        print("%d is done." % (i+1))