import torch
import os
import numpy as np
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def get_tp_fp_tn_fn(predict,gt,n_class):
    statisic = dict()
    for i in range(n_class):
        statisic[i] = dict()
        a = predict==i
        b = gt==i
        tn =  np.logical_and(a,b)
        i_a = np.zeros_like(gt)
        i_b = np.zeros_like(gt)
        i_tn = np.zeros_like(gt)
        i_a[a==True] = 1
        i_b[b==True] = 1
        i_tn[tn==True] = 1
        statisic[i]['tp']=np.sum(i_tn)
        statisic[i]['fp']=np.sum(i_b) -np.sum(i_tn)
        statisic[i]['fn']=np.sum(i_a) -np.sum(i_tn)
        statisic[i]['tn']= np.prod(predict.shape) - statisic[i]['tp'] - statisic[i]['fp']-statisic[i]['fn']
    return statisic