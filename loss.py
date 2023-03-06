import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F2

"""
IoU_loss:
    Compute IoU loss between predictions and ground-truths for training [Equation 3].
"""
class Fmeasure_loss(nn.Module):
    def __init__(self, log_like=False):
        super(Fmeasure_loss, self).__init__()
        self.beta = 0.9
        self.log_like = log_like

    def forward(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return torch.mean(floss)



def IoU_loss(preds_list, gt):
    preds = torch.cat(preds_list, dim=1)

    N, C, H, W = preds.shape
    min_tensor = torch.where(preds < gt, preds, gt)    # shape=[N, C, H, W]
    max_tensor = torch.where(preds > gt, preds, gt)    # shape=[N, C, H, W]
    min_sum = min_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    max_sum = max_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    loss = 1 - (min_sum / max_sum).mean()
    '''loss_f = nn.MSELoss()
    loss = 0.
    for pred in preds_list:
        loss += loss_f(pred, gt)
    loss = loss/len(preds_list)'''

    return loss

def BCE_loss(preds_list, gt):

    loss = 0.
    for pred in preds_list:
        loss += F2.binary_cross_entropy(pred, gt)
    loss = loss/len(preds_list)

    return loss