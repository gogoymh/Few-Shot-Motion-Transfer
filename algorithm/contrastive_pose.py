import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
#from torchlars import LARS

from contrastive_network import texture, encoder
from contrastive_loss import NTXentLoss
from data import people

########################################################################################################################
batch_size = 40
lr = 0.1
total_epoch = 1000

#path = "C://유민형//개인 연구//Few-Shot Motion Transfer//openpose//examples//output//"
path = "/home/compu/ymh/FSMT/dataset/input/"

save_name = "/home/compu/ymh/FSMT/save/pose.pth"

########################################################################################################################
#color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.ToPILImage(),
         transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
         transforms.RandomCrop(256, pad_if_needed=True),
         #transforms.RandomApply([color_jitter], p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
People = people(path, transform)
train_loader = DataLoader(People, batch_size=batch_size, shuffle=True, pin_memory=True)

########################################################################################################################
device = torch.device("cuda:0")

model = texture().to(device)
encoding = encoder().to(device)

params = list(model.parameters()) + list(encoding.parameters())
optim = optim.Adam(params, lr=lr)

contrastive_loss = NTXentLoss()

for epoch in range(total_epoch):
    running_loss = 0
    for idx, (x1, x2) in enumerate(train_loader):
        
        tmp_size = x1.shape[0]
        
        optim.zero_grad()
        
        x1 = x1.float().to(device)
        x2 = x2.float().to(device)
        
        out1 = encoding(model(x1))
        out2 = encoding(model(x2))
        
        mask = contrastive_loss._get_correlated_mask(tmp_size).to(device)
        labels = torch.zeros(2 * tmp_size).long().to(device)
        
        loss = contrastive_loss(out1, out2, mask, labels, tmp_size)
        loss.backward()
        
        optim.step()
        running_loss += loss.item()
        #print(idx, loss.item())
        
    running_loss /= len(train_loader)
    print("[Epoch:%d] [loss:%f]" % (epoch+1, running_loss))
    torch.save({'model_state_dict': model.state_dict()}, save_name)