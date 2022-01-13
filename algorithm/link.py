import torch
import torch.nn as nn

class Converter(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Linear(4096, 512)
        self.relu = nn.LeakyReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        return self.relu(self.fc(x))

class Converter1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Linear(2048, 512)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        return self.fc(x)
    
    
class Converter2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(2048, 512, 3, 1, 1, bias=False)
        self.avgpool = nn.AvgPool2d(2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        return self.avgpool(self.conv(x))
    
if __name__ == "__main__":
    
    #a = torch.randn((1,4096))
    #b = Converter1()
    #c = b(a)
    
    a = torch.randn((1,2048,8,8))
    b = Converter2()
    c = b(a)
    print(c.shape)