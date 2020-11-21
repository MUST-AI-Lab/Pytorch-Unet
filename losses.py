import torch
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import sum_tensor,softmax_helper
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','WeightBCELoss','WeightBCEDiceLoss','FocalLoss','MultiFocalLoss','SoftDiceLossV2','WeightCrossEntropyLoss',
'WeightCrossEntropyLossV2','DiceLossV3','ASLLoss','ASLLossOrigin','GDL','EqualizationLoss','FilterLoss','LogitDivCELoss','LogitAddCELoss','FilterWCELoss',
"FilterFocalLoss","MultiFocalLossV3",'EqualizationLossV2','FilterFocalLossV2','FilterCELossV2','MultiFocalLossV4']

# <--------------------------- BINARY LOSSES --------------------------->
# ================================================
class FocalLoss(nn.Module):
    def __init__(self,args, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.args = args
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds, labels,epoch,weight=None):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -1*self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

class WeightBCEDiceLoss(nn.Module):
    def __init__(self,args):
            super().__init__()
            self.args = args
            self.weight_bce = WeightBCELoss(args)

    def forward(self,output, target,epoch, weights=None):
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

    def forward(self,output, target,epoch, weights=None):
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

    def forward(self, input,epoch, target):
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

    def forward(self, input,epoch, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

# <--------------------------- MULTICLASS LOSSES --------------------------->
# ================================================

# --------------------------- dice series ---------------------------
def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
    return loss.sum() / N

class SoftDiceLossV2(nn.Module):
    __name__ = 'dice_loss'
 
    def __init__(self,args, activation='sigmoid', reduction='mean'):
        super(SoftDiceLossV2, self).__init__()
        self.args = args
        self.activation = activation
        self.num_classes = args.num_classes
 
    def forward(self, y_pred, y_true,epoch,weight=None):
        # assert weight ==None, 'SoftDiceLossV2 not yet implement  weight loss'
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce =False not suport by this Loss ")
        shp_x = y_pred.shape
        shp_y = y_true.shape
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))
        y_onehot = torch.zeros(shp_x)
        if y_pred.device.type == "cuda":
            y_onehot = y_onehot.cuda(y_true.device.index)
        y_onehot.scatter_(1, y_true, 1)
        class_dice = []
        for i in range(1, self.num_classes):
            if weight is None:
                class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_onehot[:, i:i + 1, :], activation=self.activation))
            else:
                class_dice.append(weight[0][i]*diceCoeff(y_pred[:, i:i + 1, :], y_onehot[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice
class DiceLossV3(nn.Module):
    __name__ = 'dice_loss'
    def __init__(self,args, activation='sigmoid', reduction='mean'):
        super(DiceLossV3, self).__init__()
        self.args = args
        self.activation = activation
        self.num_classes = args.num_classes

    def forward(self, y_pred, y_true,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce =False not suport by this Loss ")
        shp_x = y_pred.shape
        shp_y = y_true.shape
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))
        target = torch.zeros(shp_x)
        if y_pred.device.type == "cuda":
            target = target.cuda(y_true.device.index)
        target.scatter_(1, y_true, 1)
        logit = y_pred
        if not (target.size() == logit.size()):
            raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))
        preds = torch.sigmoid(logit)
        sum_dims = list(range(1, logit.dim()))

        if weight is None:
            dice = 2 * torch.sum(preds * target, dim=sum_dims) / torch.sum(preds ** 2 + target ** 2, dim=sum_dims)
        else:
            dice = 2 * torch.sum(weight * preds * target, dim=sum_dims) \
                / torch.sum(weight * (preds ** 2 + target ** 2), dim=sum_dims)
        loss = 1 - dice
        return loss.mean()
class ASLLoss(nn.Module):
    __name__ = 'ASLLoss'
    def __init__(self,args, activation='sigmoid', reduction='mean'):
        super(ASLLoss, self).__init__()
        self.args = args
        self.activation = activation
        self.num_classes = args.num_classes
        self.beta =1.5

    def forward(self, y_pred, y_true,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce =False not suport by this Loss ")
        shp_x = y_pred.shape
        shp_y = y_true.shape
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))
        target = torch.zeros(shp_x)
        if y_pred.device.type == "cuda":
            target = target.cuda(y_true.device.index)
        target.scatter_(1, y_true, 1)
        logit = y_pred
        if not (target.size() == logit.size()):
            raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))
        preds = torch.sigmoid(logit)
        sum_dims = list(range(1, logit.dim()))
        if weight is None:
            f_beta = (1 + self.beta ** 2) * torch.sum(preds * target, dim=sum_dims) \
                 / torch.sum(self.beta ** 2 * target ** 2 + preds ** 2, dim=sum_dims)
        else:
            f_beta = (1 + self.beta ** 2) * torch.sum(weight * preds * target, dim=sum_dims) \
                 / torch.sum(weight * (self.beta ** 2 * target ** 2 + preds ** 2), dim=sum_dims)
        loss = 1 - f_beta
        return loss.mean()
class ASLLossOrigin(nn.Module):
    __name__ = 'ASLLossOrigin'
    def __init__(self,args, activation='sigmoid', reduction='mean'):
        super(ASLLossOrigin, self).__init__()
        self.args = args
        self.activation = activation
        self.num_classes = args.num_classes
        self.beta =1.5

    def forward(self, y_pred, y_true,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce =False not suport by this Loss ")
        shp_x = y_pred.shape
        shp_y = y_true.shape
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))
        target = torch.zeros(shp_x)
        if y_pred.device.type == "cuda":
            target = target.cuda(y_true.device.index)
        target.scatter_(1, y_true, 1)
        logit = y_pred
        if not (target.size() == logit.size()):
            raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))
        preds = torch.sigmoid(logit)
        sum_dims = list(range(1, logit.dim()))
        f_beta = (1 + self.beta ** 2) * torch.sum(preds * target, dim=sum_dims) \
             / ((1 + self.beta ** 2) * torch.sum(preds * target, dim=sum_dims) +
                self.beta ** 2 * torch.sum((1 - preds) * target, dim=sum_dims) +
                torch.sum(preds * (1 - target), dim=sum_dims))
        loss = 1 - f_beta
        return loss.mean()
# Generalized Dice loss
class GDL(nn.Module):
    __name__ = 'dice_loss'
    def __init__(self,args, activation='sigmoid', reduction='mean'):
        super(GDL, self).__init__()
        self.args = args
        self.activation = activation
        self.num_classes = args.num_classes

    def forward(self, y_pred, y_true,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce =False not suport by this Loss ")
        shp_x = y_pred.shape
        shp_y = y_true.shape
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))
        target = torch.zeros(shp_x)
        if y_pred.device.type == "cuda":
            target = target.cuda(y_true.device.index)
        target.scatter_(1, y_true, 1)
        logit = y_pred
        preds = torch.sigmoid(logit)
        preds_bg = 1 - preds  # bg = background
        preds = torch.cat([preds, preds_bg], dim=1)

        target_bg = 1 - target
        target = torch.cat([target, target_bg], dim=1)

        sp_dims = list(range(2, logit.dim()))
        weight=None
        weight = 1 / (1 + torch.sum(target, dim=sp_dims) ** 2)

        generalized_dice = 2 * torch.sum(weight * torch.sum(preds * target, dim=sp_dims), dim=-1) \
                    / torch.sum(weight * torch.sum(preds ** 2 + target ** 2, dim=sp_dims), dim=-1)

        loss = 1 - generalized_dice

        return loss.mean()

# ---------------------------entropy series---------------------------
class MultiFocalLossInner(nn.Module):
    def __init__(self, args ,alpha=0.5, gamma=2, ignore_index=255,reduce=True):
        super().__init__()
        self.args = args
        self.num_class = args.num_classes
        self.alpha = None
        self.gamma = 2
        self.size_average = args.loss_reduce
        self.eps = 1e-6
        self.reduce = reduce

    def forward(self, logit, target,epoch,weight=None):
        shp_fi = target.shape
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss  try to use MultiFocalLoss")

        alpha = weight
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]
        if alpha is not None:
            alpha = alpha.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        logit = torch.softmax(logit,dim=1)
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if alpha is not None:
            if alpha.device != logpt.device:
                alpha = self.alpha.to(logpt.device)
                alpha_class = alpha.gather(0,target.view(-1))
                logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        loss = loss.view(-1,shp_fi[1],shp_fi[2])
        if self.reduce:
            loss = loss.mean()

        return loss

class MultiFocalLoss(nn.Module):
    def __init__(self, args ,alpha=0.5, gamma=2, ignore_index=255):
        super().__init__()
        self.args = args
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        logpt = -1*self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        if weight is not None:
            loss = loss *weight
        if self.args.loss_reduce:
            return loss.mean()
        else:
            return loss

class MultiFocalLossV3(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, args):
        super(MultiFocalLossV3, self).__init__()
        self.args = args
        self.num_class = args.num_classes
        self.alpha = None
        self.gamma = 2
        self.size_average = args.loss_reduce
        self.eps = 1e-6
        self.balance_index=1

        # alpha 是各样本比例，但是这在初始化的时候是未知的，所以直接跳过
        # 当设置weight的时候则可以进行加权。否则alpha默认视为1
        # if isinstance(self.alpha, (list, tuple)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.Tensor(list(self.alpha))
        # elif isinstance(self.alpha, (float,int)):
        #     assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
        #     assert  self.balance_index > -1
        #     alpha = torch.ones((self.num_class))
        #     alpha *= 1-self.alpha
        #     alpha[ self.balance_index] = self.alpha
        #     self.alpha = alpha
        # elif isinstance(self.alpha, torch.Tensor):
        #     self.alpha = self.alpha
        # else:
        #     raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target,epoch,weight=None):
        shp_fi = target.shape
        alpha = weight
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]
        if alpha is not None:
            alpha = alpha.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        logit = torch.softmax(logit,dim=1)
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if alpha is not None:
            if alpha.device != logpt.device:
                alpha = self.alpha.to(logpt.device)
                alpha_class = alpha.gather(0,target.view(-1))
                logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        loss = loss.view(-1,shp_fi[1],shp_fi[2])
        if self.size_average:
            loss = loss.mean()

        return loss

class MultiFocalLossV4(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, args):
        super(MultiFocalLossV4, self).__init__()
        self.args = args
        self.num_class = args.num_classes
        self.alpha = None
        self.gamma = 2
        self.size_average = args.loss_reduce
        self.eps = 1e-6
        self.balance_index=1

        # alpha 是各样本比例，但是这在初始化的时候是未知的，所以直接跳过
        # 当设置weight的时候则可以进行加权。否则alpha默认视为1
        # if isinstance(self.alpha, (list, tuple)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.Tensor(list(self.alpha))
        # elif isinstance(self.alpha, (float,int)):
        #     assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
        #     assert  self.balance_index > -1
        #     alpha = torch.ones((self.num_class))
        #     alpha *= 1-self.alpha
        #     alpha[ self.balance_index] = self.alpha
        #     self.alpha = alpha
        # elif isinstance(self.alpha, torch.Tensor):
        #     self.alpha = self.alpha
        # else:
        #     raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target,epoch,weight=None):
        shp_fi = target.shape
        alpha = weight
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]
        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        logit = torch.softmax(logit,dim=1)
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        # if alpha is not None:
        #     if alpha.device != logpt.device:
        #         alpha = self.alpha.to(logpt.device)
        #         alpha_class = alpha.gather(0,target.view(-1))
        #         logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        loss = loss.view(-1,shp_fi[1],shp_fi[2])
        if alpha is not None:
            if alpha.device != logpt.device:
                alpha = self.alpha.to(logpt.device)
            loss = loss*alpha

        if self.size_average:
            loss = loss.mean()
        return loss

# a warpper for cross-entropy loss
class WeightCrossEntropyLoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        loss = self.ce_fn(preds, labels)
        if weight is not None:
            loss = loss *weight
        if self.args.loss_reduce:
            return loss.mean()
        else:
            return loss
class WeightCrossEntropyLossV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reduce=True
        self.reduction="mean"

    def forward(self,input,target,epoch,weight = None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        # 这里对input所有元素求exp
        exp = torch.exp(input)
        # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
        tmp1 = exp.gather(1,target.unsqueeze(1)).squeeze()
        # 在exp第一维求和，这是softmax的分母
        tmp2 = exp.sum(1)
        # softmax公式：ei / sum(ej)
        softmax = tmp1/tmp2
        # cross-entropy公式： -yi * log(pi)
        # 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，
        # 公式中的pi就是softmax的结果
        log = -torch.log(softmax)
        if weight is not None:
            log = log * weight
        # 官方实现中，reduction有mean/sum及none
        # 只是对交叉熵后处理的差别
        if not self.reduce:
            return log
        if self.reduction == "mean": return log.mean()
        elif self.reduction == "sum": return log.sum()
        else:
            raise NotImplementedError('unkowned reduction')


# ---------------------------Logit adjustment series---------------------------
class LogitDivCELoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        if weight is not None:
            shp_weight = weight.shape
            weight = weight.view(shp_weight[0],shp_weight[1],1,1)
            preds = preds/(weight+2e-5)
        loss = self.ce_fn(preds, labels)
        if self.args.loss_reduce:
            return loss.mean()
        else:
            return loss

class LogitAddCELoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)
    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        tau = 1.0
        if weight is not None:
            shp_weight = weight.shape
            weight = weight.view(shp_weight[0],shp_weight[1],1,1)
            addition = tau * torch.log(weight)
            #addition[addition<0] = 0
            preds = preds + addition
        loss = self.ce_fn(preds, labels)
        return loss.mean()


# ---------------------------Filter series---------------------------
class  FilterFocalLoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.focal = MultiFocalLossInner(args,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        distribution = weight
        if distribution is not None:
            _filter = self.t_lambda(distribution,self.args.tail_radio)
            if preds.device.type == "cuda":
                _filter = _filter.cuda(labels.device.index)
            loss = self.focal(preds, labels)
            loss = loss *_filter
        else:
            loss = self.focal(preds, labels)
        return loss.mean()

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

class  FilterFocalLossV2(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.focal = MultiFocalLossInner(args,reduce=False)

    def get_tail_radio(self,epoch):
        if epoch <3:
            return 0.02
        elif epoch<6:
            return 0.05
        elif epoch<9:
            return 0.1
        elif epoch<12:
            return 0.17
        elif epoch<15:
            return 0.26
        elif epoch<18:
            return 0.29
        else:
            return 1.0

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        distribution = weight
        if distribution is not None:
            _filter = self.t_lambda(distribution,self.get_tail_radio(epoch))
            if preds.device.type == "cuda":
                _filter = _filter.cuda(labels.device.index)
            loss = self.focal(preds, labels,epoch)
            loss = loss *_filter
        else:
            loss = self.focal(preds, labels,epoch)
        return loss.mean()

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

class  FilterCELoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        distribution = weight
        if distribution is not None:
            ce_filter = self.t_lambda(distribution,self.args.tail_radio)
            if preds.device.type == "cuda":
                ce_filter = ce_filter.cuda(labels.device.index)
            loss = self.ce_fn(preds, labels)
            loss = loss *ce_filter
        else:
            loss = self.ce_fn(preds, labels)
        return loss.mean()

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

class  FilterCELossV2(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def get_tail_radio(self,epoch):
        if epoch <3:
            return 0.02
        elif epoch<6:
            return 0.05
        elif epoch<9:
            return 0.1
        elif epoch<12:
            return 0.17
        elif epoch<15:
            return 0.26
        elif epoch<18:
            return 0.29
        else:
            return 1.0

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        distribution = weight
        if distribution is not None:
            ce_filter = self.t_lambda(distribution,self.get_tail_radio(epoch))
            if preds.device.type == "cuda":
                ce_filter = ce_filter.cuda(labels.device.index)
            loss = self.ce_fn(preds, labels)
            loss = loss *ce_filter
        else:
            loss = self.ce_fn(preds, labels)
        return loss.mean()

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

class  FilterWCELoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        distribution = weight
        if distribution is not None:
            baseline_weight = 1 / (self.args.num_classes) * (1/(weight+2e-5))
            ce_filter = self.t_lambda(distribution,self.args.tail_radio)
            if preds.device.type == "cuda":
                ce_filter = ce_filter.cuda(labels.device.index)
            loss = self.ce_fn(preds, labels)*baseline_weight
            loss = loss *ce_filter
        else:
            loss = self.ce_fn(preds, labels)
        return loss.mean()

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

class  FilterLoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        shp_preds = preds.shape
        shp_labels = labels.shape
        if len(shp_preds) != len(shp_labels):
            labels = labels.view((shp_labels[0], 1, *shp_labels[1:]))
        onehot = torch.zeros(shp_preds)
        if preds.device.type == "cuda":
            onehot = onehot.cuda(labels.device.index)
        onehot.scatter_(1, labels, 1)

        shp_preds = preds.shape
        shp_labels = labels.shape
        if len(shp_preds) == len(shp_labels):
            labels = labels.squeeze(dim=1)
        loss = self.ce_fn(preds, labels)
        distribution = weight
        if distribution is not None:
            t_lambda = self.t_lambda(distribution,self.args.tail_radio)
            shp_t = t_lambda.shape
            t_lambda = t_lambda.view((shp_t[0], 1, *shp_t[1:]))
            if preds.device.type == "cuda":
                t_lambda = t_lambda.cuda(labels.device.index)
            new_weight =  onehot * t_lambda  
            new_weight = new_weight* (1-softmax_helper(preds))
            alpha = np.minimum(1/self.args.tail_radio,10)
            loss = alpha*loss * new_weight.sum(dim=1)
            return loss.mean()
        else:
            return loss.mean()
    
    def weight2baseline(self,weight):
        K = self.args.num_classes
        beasline_weight = 1/(K *(1/(weight+2e-5)))
        return beasline_weight

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

#this loss functio must have weight 
# see dataset weight difintion, call this loss function must have both two condition
class EqualizationLoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        shp_preds = preds.shape
        shp_labels = labels.shape
        if len(shp_preds) != len(shp_labels):
            labels = labels.view((shp_labels[0], 1, *shp_labels[1:]))
        onehot = torch.zeros(shp_preds)
        if preds.device.type == "cuda":
            onehot = onehot.cuda(labels.device.index)
        onehot.scatter_(1, labels, 1)

        shp_preds = preds.shape
        shp_labels = labels.shape
        if len(shp_preds) == len(shp_labels):
            labels = labels.squeeze(dim=1)

        distribution = weight
        if distribution is not None:
            t_lambda = self.t_lambda(distribution,self.args.tail_radio)
            shp_t = t_lambda.shape
            t_lambda = t_lambda.view((shp_t[0], 1, *shp_t[1:]))
            if preds.device.type == "cuda":
                t_lambda = t_lambda.cuda(labels.device.index)

            eql_w =1 - (1-t_lambda) * (1-onehot)
            loss =  self.ce_fn(preds,labels)
            loss = loss * eql_w
            return loss.mean()
        else:
            loss =  self.ce_fn(preds,labels)
            return loss.mean()

    def weight2baseline(self,weight):
        K = self.args.num_classes
        beasline_weight = 1/(K )*(1/(weight+2e-5))
        return beasline_weight

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

class EqualizationLossV2(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def get_tail_radio(self,epoch):
        if epoch <13:
            return 0.05
        elif epoch<18:
            return 0.1
        elif epoch<23:
            return 0.17
        else:
            return 1.0

    def forward(self, preds, labels,epoch,weight=None):
        if not self.args.loss_reduce:
            raise NotImplementedError("self.args.loss_reduce  False=not suport by this Loss ")
        shp_preds = preds.shape
        shp_labels = labels.shape
        if len(shp_preds) != len(shp_labels):
            labels = labels.view((shp_labels[0], 1, *shp_labels[1:]))
        onehot = torch.zeros(shp_preds)
        if preds.device.type == "cuda":
            onehot = onehot.cuda(labels.device.index)
        onehot.scatter_(1, labels, 1)

        shp_preds = preds.shape
        shp_labels = labels.shape
        if len(shp_preds) == len(shp_labels):
            labels = labels.squeeze(dim=1)

        distribution = weight
        if distribution is not None:
            t_lambda = self.t_lambda(distribution,self.get_tail_radio(epoch))
            shp_t = t_lambda.shape
            t_lambda = t_lambda.view((shp_t[0], 1, *shp_t[1:]))
            if preds.device.type == "cuda":
                t_lambda = t_lambda.cuda(labels.device.index)

            eql_w =1 - (1-t_lambda) * (1-onehot)
            loss =  self.ce_fn(preds,labels)
            loss = loss * eql_w
            return loss.mean()
        else:
            loss =  self.ce_fn(preds,labels)
            return loss.mean()

    def weight2baseline(self,weight):
        K = self.args.num_classes
        beasline_weight = 1/(K )*(1/(weight+2e-5))
        return beasline_weight

    def t_lambda(self,distribution,tail_radio=0.1):
        # distribution[B,H,W]
        return torch.le(distribution,tail_radio).type(torch.FloatTensor)

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)
    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2
    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)
    return tp, fp, fn, tn


