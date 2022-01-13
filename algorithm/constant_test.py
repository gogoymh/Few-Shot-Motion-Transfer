import torch
import torch.nn as nn

'''
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        print(out.shape)
        return out


a = ConstantInput(512)
b = torch.randn((4))
c = a(b)
'''

'''
a = [torch.randn((4, 512))]
styles = [s for s in a]
print(len(styles))
print(styles[0].shape)
'''
a = torch.zeros(1)
b = torch.rand((1))
print(a)
print(b)