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

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def mean_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.mean(int(ax))
    return inp

def flip(x, dim):
    """
    flips the tensor at dimension dim (mirroring!)
    :param x:
    :param dim:
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

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

def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

#获取某个文件夹下所有文件的名字
def file_name(file_dir): 
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件
        return files

#数据集使用，按行读取
def sample_array(file_name):
    f = open(file_name)               # 返回一个文件对象
    line = f.readline()               # 调用文件的 readline()方法
    ret = []
    while line:                 # 后面跟 ',' 将忽略换行符
        #print(line, end = '')　      # 在 Python 3 中使用
        ret.append(line[:-1])
        line = f.readline()
    f.close()
    return ret

def to_int(str):
    try:
        int(str)
        return int(str)
    except ValueError: #报类型错误，说明不是整型的
        try:
            float(str) #用这个来验证，是不是浮点字符串
            return int(float(str))
        except ValueError:  #如果报错，说明即不是浮点，也不是int字符串。   是一个真正的字符串
            return False

#用于在训练时上下文传递的类
class Context:
    def __init__(self):
        self.args = None
        self.net =None
        self.optimizer = None
        self.schduler = None
        self.epoch = None
        self.nonlinear = None
        self.train_loader = None
        self.device=None

def weight_norm(net,args):
    if args.deep_supervision:
        raise NotImplementedError("weight norm is not suitable for deep_supervision")
    if net.final is None:
        raise NotImplementedError("weight norm final is not found")
    final = None
    final_grad = None
    for name,parameters in net.named_parameters():
        #print(name,':',parameters.size())
        if name == 'final.weight':
            final=parameters.cpu().detach().numpy()
            final_grad=parameters.grad.cpu().detach().numpy()
    final = np.array(final)
    final = np.reshape(final,(args.num_classes,-1))
    final=np.linalg.norm(final, axis=1, keepdims=True)
    final = np.reshape(final,(args.num_classes))

    final_grad = np.array(final_grad)
    final_grad = np.reshape(final_grad,(args.num_classes,-1))
    final_grad=np.linalg.norm(final_grad, axis=1, keepdims=True)
    final_grad = np.reshape(final_grad,(args.num_classes))
    return final,final_grad

def weight_norm_init(net,args):
    if args.deep_supervision:
        raise NotImplementedError("weight norm is not suitable for deep_supervision")
    if net.final is None:
        raise NotImplementedError("weight norm final is not found")
    final = None
    for name,parameters in net.named_parameters():
        #print(name,':',parameters.size())
        if name == 'final.weight':
            final=parameters.cpu().detach().numpy()
    final = np.array(final)
    final = np.reshape(final,(args.num_classes,-1))
    final=np.linalg.norm(final, axis=1, keepdims=True)
    final = np.reshape(final,(args.num_classes))
    return final