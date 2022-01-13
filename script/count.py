import torch
import torch.nn as nn

from new_architecture33 import  Discriminator, Generator

model = Discriminator()
parameter = list(model.parameters())

cnt = 0
for i in range(len(parameter)):
    cnt += parameter[i].reshape(-1).shape[0]
    
print(cnt)
#model.to(device)
#summary(model, (3, 32, 32))