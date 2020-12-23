import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

from torch.utils.data import DataLoader, random_split
from utils.tools import AverageMeter,str2bool,softmax_helper
from sklearn.model_selection import train_test_split
from glob import glob
import collections
from utils.metrics import IOU,pixel_error,rand_error,dice_coeff,mIOU
import utils.data_vis as vis
from torch.optim import lr_scheduler
import pandas as pd
from utils.tools  import set_seed
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
from PIL import Image
# options in config
import archs
import utils.savepoints as savepoints
import losses
import utils.dataset as datasets
import matplotlib.pyplot as plt
DATASET_NAMES = datasets.__all__
ARCH_NAMES = archs.__all__
SAVE_POINTS = savepoints.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('CrossEntropyLoss')
LOSS_NAMES.append('BCEWithLogitsLoss')

# BackPropagation methods
optimizer =None
scheduler =None
criterion = None
n_train=0
n_val=0
datamaker = None


def get_args():
    # base
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment', default='default2')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-as', '--accumulation-step', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='accumulation_step')
    parser.add_argument('--seed', type=int, default=45,
                        help='a seed  for initial val', dest='seed')
    parser.add_argument('--device', default='cpu',
                        help='choose device', dest='device')
    parser.add_argument('--device_id', type=int, default=0,
                        help='a number for choose device', dest='device_id')
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='FCNNhub',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: UNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=32, type=int,
                        help='number of classes')

    # loss
    parser.add_argument('--loss', default='WeightCrossEntropyLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: CrossEntropyLoss)')
    parser.add_argument('--weight_loss', default='true', type=str2bool)
    parser.add_argument('--weight_bias', type=float, default=1e-11)
    parser.add_argument('--weight_type', default='single_baseline_weight')
    # hyper parameter for FilterLoss
    parser.add_argument('--tail_radio', type=float, default=1.0)
    parser.add_argument('--loss_reduce', default=False, type=str2bool)

    # dataset
    parser.add_argument('--dataset', metavar='DATASET', default='Cam2007DatasetV2',
                        choices=DATASET_NAMES,
                        help='model architecture: ' +
                        ' | '.join(DATASET_NAMES) +
                        ' (default: BasicDataset)')
    parser.add_argument('--data_dir', default='./data/Cam2007_n',
                        help='dataset_location_dir')
    parser.add_argument('--num_workers', default=0, type=int)
    #for dsb dataset compact
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    #for mic dataset compact
    parser.add_argument('--target_set', default='HeLa',
                        help='name')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=-1,
                        help='Downscaling factor of the images')
    #for the rate of model validation
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-100)')

    # optimizer
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD',],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    # back_up
    parser.add_argument('--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--save_check_point', dest='save_check_point', type=str2bool, default=True,
                        help='save model from a .pth file')
    parser.add_argument('--save_mode', dest='save_mode', type=str, default='by_best',choices=['by_epoch', 'by_best',],
                        help='save model from a .pth file')
    parser.add_argument('--dir_checkpoint', default='./checkpoint/',
                        help='dataset_location_dir')
    parser.add_argument('--savepoint', default='StillFalse',
                        choices=SAVE_POINTS,
                        help='model architecture: ' +
                        ' | '.join(SAVE_POINTS) +
                        ' (default: StillFalse)')
    parser.add_argument('--force_save_last', dest='force_save_last', type=str, default=False,
                        help='Load model from a .pth file')


    return parser.parse_args()

all_ord=2

def logit_norms(logits,args):
    #print(logits)
    logits = logits.cpu().detach().numpy()
    logits = np.array(logits)
    logits = np.reshape(logits,(args.num_classes,-1))
    logits_norms=np.linalg.norm(logits, ord=all_ord,axis=1, keepdims=False)
    return  logits_norms

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
    final=np.linalg.norm(final,ord=all_ord, axis=1, keepdims=False)

    final_grad = np.array(final_grad)
    final_grad = np.reshape(final_grad,(args.num_classes,-1))
    final_grad=np.linalg.norm(final_grad, ord=all_ord,axis=1, keepdims=False)
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

def get_optimizer(args,model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                              nesterov=args.nesterov, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer=optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(args,optimizer):
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                   verbose=1, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler

def get_criterion(args,model):
    if args.loss == 'CrossEntropyLoss' or args.loss == 'BCEWithLogitsLoss':
        if model.n_classes > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[args.loss](args).cuda()

    return criterion

def slice_checker(args,s_pred,gt,i):
    s_pred = softmax(s_pred,axis=0)
    t_gt = torch.from_numpy(gt).type(torch.LongTensor)
    # get onehot
    shp_preds = s_pred.shape
    shp_labels = t_gt.shape
    t_gt = t_gt.view((-1,1))
    t_onehot = torch.zeros(shp_preds[1]*shp_preds[2],shp_preds[0])
    t_onehot.scatter_(1, t_gt, 1)
    t_onehot=t_onehot.view(shp_labels[0],shp_labels[1],-1)
    t_gt = t_gt.view(shp_labels)
    #print(t_onehot.argmax(-1) == t_gt) 
    onehot = t_onehot.cpu().numpy()

    clazzes = s_pred.shape[0]
    try:
        os.mkdir('./result/{}/{}th'.format(args.experiment,i))
    except OSError:
        pass
    for item in range(clazzes):
        one_slice = s_pred[item]
        one_slice = one_slice * 255
        one_slice = Image.fromarray(one_slice.astype(np.uint8))
        one_slice_gt = onehot[:,:,item]
        one_slice_gt = one_slice_gt * 255
        one_slice_gt = Image.fromarray(one_slice_gt.astype(np.uint8))
        fig, axes = plt.subplots(1, 2, figsize=(14,7))
        axes[0].imshow(one_slice)
        axes[0].set_title("predict")
        axes[1].imshow(one_slice_gt)
        axes[1].set_title("gt")
        plt.savefig("{}/{}/{}th/{}_channel.png".format('result',args.experiment,i,item))



#main entry
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(args.device_id)
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')
    checkpoint = None
    load_from = None
    if args.load:
        load_from = args.load #备份加载位置，因为args会被替换
        checkpoint = torch.load(args.load, map_location=device)
        if 'args' in checkpoint:#兼容设置：因为旧版的部分运行保存没有保存参数args，所以有些读取是没有这个参数的 以免报错
            args = checkpoint['args']
            #delete or in gpu env
            args.device = "cpu"
            logging.info(f'''reload training:
            args:          {args}
            ''')
    #set seed
    set_seed(args.seed)
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = archs.__dict__[args.arch](args)
    optimizer = get_optimizer(args,net)
    scheduler = get_scheduler(args,optimizer)
    criterion = get_criterion(args,net)
    start_epoch =0

    #keep initial net weight norm info here
    weight_norms = weight_norm_init(net,args)
    init_norms = OrderedDict()
    for idx in range(args.num_classes):
        init_norms['init_norm_{}'.format(idx)] = weight_norms[idx]

    #mkdirs for each experiment
    try:
        os.mkdir('./result/{}'.format(args.experiment))
    except OSError:
        pass
    print(net)
    logging.info(f'Network:\n'
                 f'\t{args.input_channels} input channels\n'
                 f'\t{args.num_classes} output channels (classes)\n')
    if load_from is not None:
        if 'args' in checkpoint:#新版保存了 网络状态，优化器状态等，旧版没有，作兼容
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.device == "cuda":
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            start_epoch = checkpoint['epoch']
            logging.info(f'Model loaded from {load_from} in epoch {start_epoch}')
        else:
            net.load_state_dict(checkpoint)

    #check the load weight norm
    load_weight_norms = weight_norm_init(net,args)
    final_norms = OrderedDict()
    for idx in range(args.num_classes):
        final_norms['final_norm_{}'.format(idx)] = load_weight_norms[idx]

    net.to(device=device)

    # faster convolutions, but more memory
    # cudnn.benchmark = True
    # init data set here:
    #global datamaker
    datamaker = datasets.__dict__[args.dataset](args)
    train_loader,val_loader,n_train,n_val = datamaker(args)
    logging.info(f'''Starting training:
        args:          {args}
    ''')

    #for csv
    log = OrderedDict()
    #init csv files
    for i in range(args.num_classes):
        log['weight_norm_{}'.format(i)] = []
        log['gradient_norm_{}'.format(i)] = []
        log['logit_norm_{}'.format(i)] = []
        log['iou_{}'.format(i)] = []


    net.train() #因为要获得对应的损失梯度，所以需要打开训练开关
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    batch_count = 0
    avg_meters = {'loss': AverageMeter(),'miou': AverageMeter()}
    epoch=args.epochs #这里是最后一个epoch

    #测试验证集的图片，获取训练结果的分布
    with tqdm(total=n_val, desc='Validation round') as pbar:
        show=False
        for batch  in val_loader:
            imgs = batch['image']
            true_masks = batch['mask']
            if 'weight' in batch:
                weight = batch['weight']
            else:
                weight = None
            imgs = imgs.to(device=device, dtype=torch.float32)
            if weight is not None:
                weight = weight.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            mask_pred = net(imgs)
            if args.deep_supervision: #choose final
                mask_pred = mask_pred[-1]

            if net.n_classes > 1:
                loss = criterion(mask_pred, true_masks,epoch)
                if not args.loss_reduce:
                    loss = loss.mean()
                #----------------------------check the final round
                optimizer.zero_grad()
                loss.backward()


                avg_meters['loss'].update(loss.cpu().item())
                pbar.set_postfix(**{'val_loss': avg_meters['loss'].avg})
                for i in range(imgs.shape[0]):
                    #---------------------------- for norms
                    weight_norms,gradient_norms = weight_norm(net,args) #collect weight norm for final layer
                    for clazz in range(args.num_classes):
                        log['weight_norm_{}'.format(clazz)].append(weight_norms[clazz])
                        log['gradient_norm_{}'.format(clazz)].append(gradient_norms[clazz])
                    #----------------------------for iou
                    s_true_mask =  true_masks[i].cpu().detach().numpy()
                    s_pred = mask_pred[i].cpu().detach().numpy()
                    img = imgs[i].cpu().detach().numpy()
                    #-----------------------------------------------for slice each channel
                    slice_checker(args,s_pred.copy(),s_true_mask.copy(),"{}_{}".format(batch_count,i))
                    #-----------------------------------------------
                    s_pred = np.argmax(s_pred,axis=0)
                    if True:
                        datamaker.showrevert_cp2file(img,s_true_mask,s_pred,args.experiment,'{}_{}th'.format(batch_count,epoch))
                        show = True
                    miou,statisic= mIOU(s_pred,s_true_mask,net.n_classes)
                    for key in statisic:
                        iou = (statisic[key]['tp']*1.0) / (statisic[key]['tp']+statisic[key]['fp']+statisic[key]['fn']+(-1e-5))
                        log['iou_{}'.format(key)].append(iou)
                    #--------------------------for logits
                    softmax_logits = softmax_helper(mask_pred[i].unsqueeze(0))
                    norms = logit_norms(mask_pred[i],args)
                    for clazz in range(args.num_classes):
                        log['logit_norm_{}'.format(clazz)].append(norms[clazz])
                else:
                    pass #not implment here
                    # pred = torch.sigmoid(mask_pred)
                    # pred_int = (pred > 0.5).int()
                    # pred = (pred > 0.5).float()
                    # if args.weight_loss:
                    #     loss = criterion(mask_pred, true_masks,epoch,weight)
                    # else:
                    #     loss = criterion(mask_pred, true_masks,epoch)
                    # avg_meters['loss'].update(loss.cpu().item())
                    # # for i in range(imgs.shape[0]):
                    # for i in range(imgs.shape[0]):
                    #     s_true_mask =  true_masks[i].cpu().detach().numpy()
                    #     s_pred = pred_int[i].cpu().detach().numpy()
                    #     if not show:# once for a epoch
                    #         if i==0:
                    #             vis.visualize_pred_to_file("./result/{}/epoch:{}_{}_{}.png".format(args.experiment,epoch,batch_count,i),imgs[i].cpu().detach().numpy(), s_true_mask, s_pred , title1="Original", title2="True", title3="Pred[0]")
                    #             stack = 'val_loss:{},iou:{},pixel_error:{},rand_error:{},dice_coeff:{}'.format(loss.cpu().item(),pixel_error(s_true_mask,s_pred),IOU(s_true_mask,s_pred),rand_error(s_true_mask,s_pred),dice_coeff(pred, true_masks).item())
                    #             with open("./result/{}/epoch:{}_{}_{}.txt".format(args.experiment,epoch,batch_count,i),'w') as f:    #设置文件对象
                    #                 f.write(stack)
                    #         show = True
                    #     avg_meters['pixel_error'].update(pixel_error(s_true_mask,s_pred))
                    #     avg_meters['iou'].update(IOU(s_true_mask,s_pred))
                    #     avg_meters['rand_error'].update(rand_error(s_true_mask,s_pred))
                    #     avg_meters['dice_coeff'].update(dice_coeff(pred, true_masks).item())
                    #     pbar.set_postfix(**{'val_loss': avg_meters['loss'].avg,'iou': avg_meters['iou'].avg,'pixel_error': avg_meters['pixel_error'].avg,'rand_error': avg_meters['rand_error'].avg})
            pbar.update(imgs.shape[0])
            batch_count+=1
        pbar.close()
        pd.DataFrame(log).to_csv('./result/{}.csv'.format(args.experiment),index=None)
