import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class Random(Dataset):
    def __init__(self):
        super().__init__()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
        
        self.len = 4800
        
    def __getitem__(self, index):
        
        img = np.random.choice(255, size=(28,28,1), replace=True)#.astype('float32')
        img = img / 255
        
        img = self.transform(img)
        
        
        return img
    
    def __len__(self):
        
        return self.len


from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(28*28))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *(1,28,28))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(28*28)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

#adversarial_loss = torch.nn.CrossEntropyLoss()
adversarial_loss = torch.nn.BCELoss()

device = torch.device("cuda:0")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

batch_size = 128

dataloader = DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

next_discriminator1 = Discriminator().to(device)
optimizer_N1 = torch.optim.Adam(next_discriminator1.parameters(), lr=0.0002, betas=(0.5, 0.999))

next_discriminator2 = Discriminator().to(device)
optimizer_N2 = torch.optim.Adam(next_discriminator2.parameters(), lr=0.0002, betas=(0.5, 0.999))

next_discriminator3 = Discriminator().to(device)
optimizer_N3 = torch.optim.Adam(next_discriminator3.parameters(), lr=0.0002, betas=(0.5, 0.999))
'''
first = torch.zeros((batch_size)).long().to(device)
second = torch.ones((batch_size)).long().to(device)
third = 2 * torch.ones((batch_size)).long().to(device)
forth = 3 * torch.ones((batch_size)).long().to(device)
fifth = 4 * torch.ones((batch_size)).long().to(device)
'''
zeros = torch.zeros((batch_size,1)).float().to(device)
ones = torch.ones((batch_size,1)).float().to(device)

random_set = Random()
random_loader = DataLoader(random_set, batch_size=batch_size, shuffle=True)

for epoch in range(200):
    for i, (imgs, _) in enumerate(dataloader):
        
        imgs, _ = dataloader.__iter__().next()
        imgs = imgs.float().to(device)
        z = torch.from_numpy(np.random.normal(0, 1, (batch_size, 100))).float().to(device)
        
        gen_imgs = generator(z)
        ## -------- ##
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        generator.eval()
        discriminator.train()
        
        real_loss = adversarial_loss(discriminator(imgs), ones)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), zeros)
        
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        
        ## -------- ##
        optimizer_G.zero_grad()
        optimizer_N1.zero_grad()
        generator.eval()
        next_discriminator1.train()
        
        next_discriminator1.load_state_dict(discriminator.state_dict())
        real_loss = adversarial_loss(next_discriminator1(imgs), ones)
        fake_loss = adversarial_loss(next_discriminator1(gen_imgs.detach()), zeros)
        
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_N1.step()
        
        ## -------- ##
        optimizer_G.zero_grad()
        optimizer_N2.zero_grad()
        generator.eval()
        next_discriminator2.train()
        
        next_discriminator2.load_state_dict(next_discriminator1.state_dict())
        real_loss = adversarial_loss(next_discriminator2(imgs), ones)
        fake_loss = adversarial_loss(next_discriminator2(gen_imgs.detach()), zeros)
        
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_N2.step()
        
        ## -------- ##
        optimizer_G.zero_grad()
        optimizer_N3.zero_grad()
        generator.eval()
        next_discriminator3.train()
        
        next_discriminator3.load_state_dict(next_discriminator2.state_dict())
        real_loss = adversarial_loss(next_discriminator3(imgs), ones)
        fake_loss = adversarial_loss(next_discriminator3(gen_imgs.detach()), zeros)
        
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_N3.step()
        
        ## -------- ##
        optimizer_G.zero_grad()
        generator.train()
        discriminator.eval()

        gen_imgs = generator(z)

        g_loss = adversarial_loss(next_discriminator3(gen_imgs), ones)
        
        g_loss.backward()
        optimizer_G.step()
        
        

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, 200, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % 400 == 0:
            save_image(gen_imgs.data[:25], "images8/%d.png" % batches_done, nrow=5, normalize=True)