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

from torch.utils.data import DataLoader, random_split
from utils.tools import AverageMeter,str2bool
from sklearn.model_selection import train_test_split
from glob import glob
import collections
from utils.metrics import IOU,pixel_error,rand_error,dice_coeff
import utils.data_vis as vis
from torch.optim import lr_scheduler
import pandas as pd
from utils.tools  import set_seed
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

# options in config
import archs
import utils.savepoints as savepoints
import losses
import utils.dataset as datasets
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


def get_args():
    # base
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('--continue_epochs', type=int, default=-1,
                        help='resume from a checkpoint epochs', dest='continue_epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('--seed', type=int, default=45,
                        help='a seed  for initial val', dest='seed')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='FCNN',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: UNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')

    # loss
    parser.add_argument('--loss', default='WeightBCELoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: WeightBCELoss)')
    parser.add_argument('--weight_loss', default=False, type=str2bool)
    parser.add_argument('--weight_bias', type=float, default=1e-11)

    # dataset
    parser.add_argument('--dataset', metavar='DATASET', default='MICDataset',
                        choices=DATASET_NAMES,
                        help='model architecture: ' +
                        ' | '.join(DATASET_NAMES) +
                        ' (default: BasicDataset)')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    parser.add_argument('--data_dir', default='./data/mic/',
                        help='dataset_location_dir')
    parser.add_argument('--target_set', default='HeLa',
                        help='name')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=-1,
                        help='Downscaling factor of the images')
    parser.add_argument('--num_workers', default=0, type=int)

    #validation
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
    parser.add_argument('--savepoint', default='AVG',
                        choices=SAVE_POINTS,
                        help='model architecture: ' +
                        ' | '.join(SAVE_POINTS) +
                        ' (default: AVG)')
    parser.add_argument('--force_save_last', dest='force_save_last', type=str, default=False,
                        help='Load model from a .pth file')


    return parser.parse_args()

def train_net(net,device,train_loader,args):
    net.train()
    avg_meters = {'loss': AverageMeter()}

    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
        for batch in train_loader:
            imgs = batch['image']
            true_masks = batch['mask']
            weight = batch['weight']
            assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            weight = weight.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            masks_pred = net(imgs)
            # compute output
            if args.deep_supervision:
                loss = 0
                for output in masks_pred:
                    if args.weight_loss:
                        loss += criterion(output, true_masks,weight)
                    else:
                        loss += criterion(output, true_masks)
                loss /= len(masks_pred)
            else:
                if args.weight_loss:
                    loss = criterion(masks_pred, true_masks,weight)
                else:
                    loss = criterion(masks_pred, true_masks)

            avg_meters['loss'].update(loss.cpu().item())
            pbar.set_postfix(**{'train_loss': avg_meters['loss'].avg})

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            pbar.update(imgs.shape[0])
        pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg)])

def eval_net(net, device, val_loader ,args):
    """Evaluation without the densecrf with the dice coefficient"""
    avg_meters = {'loss': AverageMeter(),'iou': AverageMeter(),'pixel_error': AverageMeter(),'rand_error': AverageMeter(),'dice_coeff':AverageMeter()}
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    batch_count = 0

    with tqdm(total=n_val, desc='Validation round') as pbar:
        for batch  in val_loader:
            imgs = batch['image']
            true_masks = batch['mask']
            weight = batch['weight']
            imgs = imgs.to(device=device, dtype=torch.float32)
            weight = weight.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if args.deep_supervision: #choose final
                mask_pred = mask_pred[-1]

            if net.n_classes > 1:#need to fix
                loss = F.cross_entropy(mask_pred, true_masks).item()
                avg_meters['loss'].update(loss.cpu().item())
            else:
                pred = torch.sigmoid(mask_pred)
                pred_int = (pred > 0.5).int()
                pred = (pred > 0.5).float()
                if args.weight_loss:
                    loss = criterion(mask_pred, true_masks,weight)
                else:
                    loss = criterion(mask_pred, true_masks)

                avg_meters['loss'].update(loss.cpu().item())
                # for i in range(imgs.shape[0]):
                for i in range(imgs.shape[0]):
                    s_true_mask =  true_masks[i].cpu().detach().numpy()
                    s_pred = pred_int[i].cpu().detach().numpy()
                    if i==0:
                        vis.visualize_pred_to_file("./result/{}/epoch:{}_{}_{}.png".format(args.experiment,epoch,batch_count,i),imgs[i].cpu().detach().numpy(), s_true_mask, s_pred , title1="Original", title2="True", title3="Pred[0]")
                        stack = 'val_loss:{},iou:{},pixel_error:{},rand_error:{},dice_coeff:{}'.format(loss.cpu().item(),pixel_error(s_true_mask,s_pred),IOU(s_true_mask,s_pred),rand_error(s_true_mask,s_pred),dice_coeff(pred, true_masks).item())
                        with open("./result/{}/epoch:{}_{}_{}.txt".format(args.experiment,epoch,batch_count,i),'w') as f:    #设置文件对象
                            f.write(stack)
                    avg_meters['pixel_error'].update(pixel_error(s_true_mask,s_pred))
                    avg_meters['iou'].update(IOU(s_true_mask,s_pred))
                    avg_meters['rand_error'].update(rand_error(s_true_mask,s_pred))
                    avg_meters['dice_coeff'].update(dice_coeff(pred, true_masks).item())

            pbar.set_postfix(**{'val_loss': avg_meters['loss'].avg,'iou': avg_meters['iou'].avg,'pixel_error': avg_meters['pixel_error'].avg,'rand_error': avg_meters['rand_error'].avg})
            pbar.update(imgs.shape[0])
            batch_count+=1
        pbar.close()

    net.train()
    ret = OrderedDict()
    ret['loss'] = avg_meters['loss'].avg
    ret['iou'] = avg_meters['iou'].avg
    ret['pixel_error'] = avg_meters['pixel_error'].avg
    ret['rand_error'] = avg_meters['rand_error'].avg
    ret['dice_coeff'] = avg_meters['dice_coeff'].avg
    return ret

def get_dataset(args):
    dataset = datasets.__dict__[args.dataset](target_set=args.target_set,
                                           data_path=args.data_dir,scale=args.scale)
    n_val = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return train_loader,val_loader,n_train,n_val

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #set seed 
    set_seed(args.seed)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = archs.__dict__[args.arch](args)
    #BackPropagation method
    optimizer = get_optimizer(args,net)
    scheduler = get_scheduler(args,optimizer)
    criterion = get_criterion(args,net)

    #mkdirs for centain experiment
    try:
        os.mkdir('./result/{}'.format(args.experiment))
    except OSError:
        pass

    print(net)
    logging.info(f'Network:\n'
                 f'\t{args.input_channels} input channels\n'
                 f'\t{args.num_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # init data set here: there two kinds of  data set mic and dsb_2018_96
    datamaker = datasets.__dict__[args.dataset](args)
    train_loader,val_loader,n_train,n_val = datamaker(args)
    logging.info(f'''Starting training:
        args:          {args}
    ''')

    #for csv
    log = OrderedDict()

    writer = SummaryWriter(comment=f'_ex.{args.experiment}_{args.optimizer}_LR_{args.lr}_BS_{args.batchsize}_model_{args.arch}')
    savepoint=savepoints.__dict__[args.savepoint](args)
    try:
        if args.continue_epochs <0:
            rounds = range(args.epochs)
        else:
            rounds = range(args.continue_epochs,args.epochs)
        for epoch in rounds:
            # train
            train_log = train_net(net=net,device=device,train_loader=train_loader,args=args)
            #validate
            val_log = eval_net(net=net,device=device,val_loader=val_loader,args=args)

            if scheduler is not None and args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])
            else:
                scheduler.step()

            if 'epoch' not in log:
                log['epoch'] = []
            log['epoch'].append(epoch+1)

            #record for tensorboard
            for m_key in train_log:
                scale = '{}/train'.format(m_key)
                writer.add_scalar(scale,train_log[m_key], (epoch+1))
                #for single file in csv
                if '{}_{}'.format('train',m_key) not in log:
                    log['{}_{}'.format('train',m_key)]=[]
                log['{}_{}'.format('train',m_key)].append(train_log[m_key])

            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), (epoch+1))
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), (epoch+1))
            logging.info('==================================================>')
            if net.n_classes > 1:
                for m_key in val_log:
                    scale = '{}/test'.format(m_key)
                    logging.info('{} : {}'.format(m_key,val_log[m_key]))
                    writer.add_scalar(scale,val_log[m_key], (epoch+1))
                    #for single file in csv
                    if '{}_{}'.format('val',m_key) not in log:
                        log['{}_{}'.format('val',m_key)]=[]
                    log['{}_{}'.format('val',m_key)].append(val_log[m_key])
            else:
                for m_key in val_log:
                    scale = '{}/test'.format(m_key)
                    logging.info('{} : {}'.format(m_key,val_log[m_key]))
                    writer.add_scalar(scale,val_log[m_key], (epoch+1))
                    #for single file in csv
                    if '{}_{}'.format('val',m_key) not in log:
                        log['{}_{}'.format('val',m_key)]=[]
                    log['{}_{}'.format('val',m_key)].append(val_log[m_key])
            logging.info('==================================================>\n')
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], (epoch+1))

            #force to save check point at last epoch
            if args.force_save_last and epoch == args.epochs-1:
                    torch.save(net.state_dict(),args.dir_checkpoint + f'CP_ex.{args.experiment}_epoch{epoch + 1}_{args.arch}_{args.dataset}.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')
            # save check point by condition
            if args.save_check_point:
                if args.save_mode == 'by_best':
                    if savepoint.is_new_best(val_log=val_log):
                        try:
                            os.mkdir(args.dir_checkpoint)
                            logging.info('Created checkpoint directory')
                        except OSError:
                            pass
                        torch.save(net.state_dict(),args.dir_checkpoint + f'CP_ex.{args.experiment}_epoch{epoch + 1}_{args.arch}_{args.dataset}.pth')
                        logging.info(f'Checkpoint {epoch + 1} saved !')
                        # for csv
                        if 'is_best' not in log:
                            log['is_best'] = []
                        log['is_best'] .append(1)
                    else:
                        if 'is_best' not in log:
                            log['is_best'] = []
                        log['is_best'] .append(0)
                else:
                    try:
                        os.mkdir(args.dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(net.state_dict(),args.dir_checkpoint + f'CP_ex.{args.experiment}_epoch{epoch + 1}_{args.arch}_{args.dataset}.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')
        #for csv
        pd.DataFrame(log).to_csv('./result/{}.csv'.format(args.experiment),index=None)
        writer.close()
    except KeyboardInterrupt:
        if args.save_check_point:
            torch.save(net.state_dict(), args.dir_checkpoint+f'INTERRUPTED_ex.{args.experiment}_epoch{epoch + 1}_{args.arch}_{args.dataset}.pth')
            logging.info('Saved interrupt')
        writer.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
