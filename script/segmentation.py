import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        
        smooth = 0
        # m1=pred.flatten()
        # m2=target.flatten()
        # intersection = (m1 * m2)

        # score=1-((2. * torch.sum(intersection) + smooth) / (torch.sum(m1) + torch.sum(m2) + smooth))
        # #score=1-((2. * torch.sum(intersection) + smooth) / (torch.sum(m1*m1) + torch.sum(m2*m2) + smooth))
                
        num = target.shape[0]
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        intersection=torch.mul(m1,m2)
        score = 1-torch.sum((2. * torch.sum(intersection,dim=1) + smooth) / (torch.sum(m1,dim=1) + torch.sum(m2,dim=1) + smooth))/num
        
        # for squared
        ## score = 1-torch.sum((2. * torch.sum(intersection,dim=1) + smooth) / (torch.sum(m1*m1,dim=1) + torch.sum(m2*m2,dim=1) + smooth))/num
        
        return score

def create_upconv(in_channels, out_channels, size=None, kernel_size=None, stride=None, padding=None):
    return nn.Sequential(
        #nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        nn.Upsample(size=size, mode='nearest')
        , nn.Conv2d(in_channels,out_channels,3,1,1)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.ReLU(inplace=True)
        , nn.Conv2d(out_channels,out_channels,3,1,1)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.ReLU(inplace=True)
        )

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_l1 = nn.Sequential(
            nn.Conv2d(3,32,3,1,1)
            , nn.BatchNorm2d(num_features=32)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(32,32,3,1,1)
            , nn.BatchNorm2d(num_features=32)
            , nn.ReLU(inplace=True)
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1)
            , nn.BatchNorm2d(num_features=64)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(64,64,3,1,1)
            , nn.BatchNorm2d(num_features=64)
            , nn.ReLU(inplace=True)
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1)
            , nn.BatchNorm2d(num_features=128)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(128,128,3,1,1)
            , nn.BatchNorm2d(num_features=128)
            , nn.ReLU(inplace=True)
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1)
            , nn.BatchNorm2d(num_features=256)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(256,256,3,1,1)
            , nn.BatchNorm2d(num_features=256)
            , nn.ReLU(inplace=True)
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1)
            , nn.BatchNorm2d(num_features=512)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(512,512,3,1,1)
            , nn.BatchNorm2d(num_features=512)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u4 = create_upconv(in_channels=512, out_channels=256, size=(40,24))

        self.conv_u4 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1)
            , nn.BatchNorm2d(num_features=256)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(256,256,3,1,1)
            , nn.BatchNorm2d(num_features=256)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u3 = create_upconv(in_channels=256, out_channels=128, size=(80,48))

        self.conv_u3 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1)
            , nn.BatchNorm2d(num_features=128)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(128,128,3,1,1)
            , nn.BatchNorm2d(num_features=128)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u2 = create_upconv(in_channels=128, out_channels=64, size=(160,96))

        self.conv_u2 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1)
            , nn.BatchNorm2d(num_features=64)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(64,64,3,1,1)
            , nn.BatchNorm2d(num_features=64)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u1 = create_upconv(in_channels=64, out_channels=32, size=(320,192))

        self.conv_u1 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1)
            , nn.BatchNorm2d(num_features=32)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(32,32,3,1,1)
            , nn.BatchNorm2d(num_features=32)
            , nn.ReLU(inplace=True)
            )

        self.conv1x1_out = nn.Conv2d(32, 1, 1, 1, 0, bias=True)
        self.smout = nn.Sigmoid()
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        out = self.conv1x1_out(output9)
        
        return self.smout(out)