import torch
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','WeightBCELoss','WeightBCEDiceLoss']

class WeightBCEDiceLoss(nn.Module):
    def __init__(self,args):
            super().__init__()
            self.args = args
            self.weight_bce = WeightBCELoss(args)

    def forward(self,output, target, weights=None):
        bce = self.weight_bce(output, target, weights)
        dice = self.get_dice(output,target)
        return 0.5 * bce + dice

    def get_dice(self,input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice


class WeightBCELoss(nn.Module):
    def __init__(self,args):
            super().__init__()
            self.args = args

    def forward(self,output, target, weights=None):
        output = torch.sigmoid(output)
        flatten_out =output.contiguous().view(-1)
        flatten_target = target.contiguous().view(-1)
        if weights is not None:
            flatten_weights = weights.contiguous().view(-1)
        if weights is not None:
            assert weights.shape==target.shape
            bias = self.args.weight_bias
            loss = flatten_target * torch.log(flatten_out+bias) + (1 - flatten_target) * torch.log(1 - flatten_out+bias)
            loss = loss*flatten_weights
        else:
            bias = self.args.weight_bias
            loss = flatten_target * torch.log(flatten_out+bias) + (1 - flatten_target) * torch.log(1 - flatten_out+bias)

        loss = torch.mean(loss)
        return torch.neg(loss)

class BCEDiceLoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss



