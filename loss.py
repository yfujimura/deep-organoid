import torch
from torch import nn

class SC_inv(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        z = torch.log(pred+1e-8)-torch.log(target+1e-8)
        sum_z = torch.sum(z)
        sum_z_2 = torch.sum(z*z)
        n = z.shape[2]*z.shape[3]
        loss = torch.sqrt( sum_z_2 / n - sum_z*sum_z/(n*n) )
        return loss
    
class CosineSimilarityLoss(nn.Module):
    def  __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        
    def forward(self, pred, target):
        _, _, height, width = pred.shape
        loss = torch.mean(1 - self.cos(pred.view(-1, height*width), target.view(-1, height*width)))
        return loss
    
class MSEandCosineSimilarityLoss(nn.Module):
    def  __init__(self, gamma=1., lmd=1.):
        super().__init__()
        self.gamma = gamma
        self.lmd = lmd
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        _, _, height, width = pred.shape
        mse = self.mse(pred, target)
        cossim = torch.mean(1 - self.cos(pred.view(-1, height*width), target.view(-1, height*width)))
        loss = self.lmd * mse + self.gamma * cossim
        return loss
    
class AdaptiveMSEandCosineSimilarityLoss(nn.Module):
    def  __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.mse = nn.MSELoss()
        self.weight = nn.Linear(1,2, bias=False)
        self.one = torch.ones(1,1).to("cuda")
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, pred, target):
        _, _, height, width = pred.shape
        mse = self.mse(pred, target)
        cossim = torch.mean(1 - self.cos(pred.view(-1, height*width), target.view(-1, height*width)))
        
        weight = self.weight(self.one) # 1 x 2
        
        loss = torch.exp(-weight[0,0]) * mse + torch.exp(-weight[0,1]) * cossim + torch.sum(weight)
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, pred, target):
        loss = torch.mean(-target * torch.pow((1-pred), self.gamma) * torch.log(pred + 1e-8) - (1-target) * torch.pow(pred, self.gamma) * torch.log(1-pred + 1e-8))
        return loss
    
def sc_inv(pred, target):
    z = torch.log(pred+1e-8)-torch.log(target+1e-8)
    sum_z = torch.sum(z)
    sum_z_2 = torch.sum(z*z)
    n = z.shape[2]*z.shape[3]
    loss = torch.sqrt( sum_z_2 / n - sum_z*sum_z/(n*n) )
    return loss

