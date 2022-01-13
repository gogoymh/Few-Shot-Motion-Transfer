import torch
import torch.nn as nn
import numpy as np

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()

        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def simililarity_function(self, x, y):
        v = self.similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    
    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    def forward(self, zis, zjs, mask, labels, batch_size):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.simililarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        #labels = torch.zeros(2 * batch_size).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batch_size = 6
    
    a = torch.randn((batch_size, 10)).to(device)
    b = torch.randn((batch_size, 10)).to(device)
    
    contrastive_loss = NTXentLoss()
    
    mask = contrastive_loss._get_correlated_mask(batch_size).to(device)
    labels = torch.zeros(2 * batch_size).long().to(device)
    
    loss = contrastive_loss(a, b, mask, labels, batch_size)
    print(loss)
    