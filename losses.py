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
'WeightCrossEntropyLossV2','DiceLossV3','ASLLoss','ASLLossOrigin','GDL','EqualizationLoss','FilterLoss']

# --------------------------- BINARY LOSSES ---------------------------
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

    def forward(self, preds, labels,weight=None):
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

# --------------------------- MULTICLASS LOSSES ---------------------------
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
 
    def forward(self, y_pred, y_true,weight=None):
        # assert weight ==None, 'SoftDiceLossV2 not yet implement  weight loss'
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

    def forward(self, y_pred, y_true,weight=None):
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

    def forward(self, y_pred, y_true,weight=None):
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

    def forward(self, y_pred, y_true,weight=None):
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

    def forward(self, y_pred, y_true,weight=None):
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
class MultiFocalLoss(nn.Module):
    def __init__(self, args ,alpha=0.5, gamma=2, ignore_index=255):
        super().__init__()
        self.args = args
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,weight=None):
        logpt = -1*self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        if weight is not None:
            loss = loss *weight
        return loss.mean()
# a warpper for cross-entropy loss
class WeightCrossEntropyLoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,weight=None):
        loss = self.ce_fn(preds, labels)
        if weight is not None:
            loss = loss *weight
        return loss.mean()
class WeightCrossEntropyLossV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reduce=True
        self.reduction="mean"

    def forward(self,input,target,weight = None):
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


class  FilterLoss(nn.Module):
    def __init__(self, args,ignore_index=255):
        super().__init__()
        self.args = args
        self.reduce=True
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss( ignore_index=self.ignore_index,reduce=False)

    def forward(self, preds, labels,weight=None):
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
        self.epoch_iter = 100 / (self.args.accumulation_step*self.args.batchsize)
        self.count_iter =0

    def forward(self, preds, labels,weight=None):
        self.count_iter  += 1
        epoch=(self.count_iter/self.epoch_iter)+1
        tail_radio =0.30* (1/epoch)**0.8
        k = 1/epoch

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
            inserve_weight = self.weight2baseline(weight)
            wce = inserve_weight*loss
            t_lambda = self.t_lambda(distribution,tail_radio)
            shp_t = t_lambda.shape
            t_lambda = t_lambda.view((shp_t[0], 1, *shp_t[1:]))
            if preds.device.type == "cuda":
                t_lambda = t_lambda.cuda(labels.device.index)
            new_weight =  onehot * t_lambda  
            new_weight = new_weight* (1-softmax_helper(preds))
            loss = loss * new_weight.sum(dim=1)
            loss = k*wce+(1-k)*loss
            return loss.mean()
        else:
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


