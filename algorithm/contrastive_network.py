import torch
import torch.nn as nn

from resnet import resnet50

class Feature(nn.Module):
    def __init__(self):
        super().__init__()
        
        a = resnet50()
        self.feature = nn.Sequential(*list(a.children())[:-1])
        
    def forward(self, x):
        x =  self.feature(x)
        x = torch.flatten(x, 1)
        return x


class Feature1(nn.Module):
    def __init__(self):
        super().__init__()
        
        a = resnet50()
        self.feature = nn.Sequential(*list(a.children())[:-1])
        
    def forward(self, x):
        x =  self.feature(x)
        x = torch.flatten(x, 1)
        return x
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(2048, 1000)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(1000, 128)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Feature2(nn.Module):
    def __init__(self):
        super().__init__()
        
        a = resnet50()
        self.feature = nn.Sequential(*list(a.children())[:-2])
        
    def forward(self, x):
        x =  self.feature(x)
        return x


if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #a = torch.randn((1,3,256,256)).to(device)
    #oper = Feature2().to(device)
    #b = oper(a)
    #print(b.shape)
    
    #oper2 = Encoder().to(device)
    #c = oper2(b)
    #print(c.shape)
    
    model = Feature()
    parameter = list(model.parameters())
    
    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
        
    print(cnt)