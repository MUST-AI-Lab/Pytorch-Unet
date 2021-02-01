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
from utils.tools import AverageMeter,str2bool,softmax_helper
from sklearn.model_selection import train_test_split
from glob import glob
import collections
from utils.metrics import IOU,pixel_error,rand_error,dice_coeff,mIOU
import utils.data_vis as vis
from torch.optim import lr_scheduler
import pandas as pd
from utils.tools  import set_seed,Context
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

from utils.tools import weight_norm,weight_norm_init

# options in config
import archs
import utils.savepoints as savepoints
import losses
import trainers
import utils.dataset as datasets
DATASET_NAMES = datasets.__all__
ARCH_NAMES = archs.__all__
SAVE_POINTS = savepoints.__all__
TRAINERS_NAMES = trainers.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('CrossEntropyLoss')
LOSS_NAMES.append('BCEWithLogitsLoss')

context=Context
# BackPropagation methods
context.optimizer =None
context.scheduler =None
context.criterion = None
context.n_train=0
context.n_val=0
context.datamaker = None


def get_args():
    # base
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment', default='default')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
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
    parser.add_argument('--trainer', default='GradientTraceTrainer',
                        choices=TRAINERS_NAMES,
                        help='trainer: ' +
                        ' | '.join(TRAINERS_NAMES) +
                        ' (default: STDTrainer)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=32, type=int,
                        help='number of classes')

    # loss
    parser.add_argument('--loss', default='SeeSawLossV3',
                        choices=LOSS_NAMES,
                        help='loss: ' + 
                        ' | '.join(LOSS_NAMES) +
                        ' (default: WeightBCELoss)')
    parser.add_argument('--weight_loss', default='true', type=str2bool)
    parser.add_argument('--weight_bias', type=float, default=1e-11)
    parser.add_argument('--weight_type', default='single_count')
    # hyper parameter for FilterLoss
    parser.add_argument('--tail_radio', type=float, default=0.05)
    parser.add_argument('--loss_reduce', default=True, type=str2bool)
    
    # for trainer
    # cb loss 
    parser.add_argument('--beta', type=float, default=0.9999)

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
    parser.add_argument('--input_w', default=696, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=520, type=int,
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
    parser.add_argument('--force_save_last', dest='force_save_last', type=str, default=True,
                        help='Load model from a .pth file')
    parser.add_argument('--no_replace',type=str,default='')
    parser.add_argument('--continnue', type=str, default=False,
                        help='continnue to train')


    return parser.parse_args()

def eval_net(context,epoch,miou_split=True):
    """Evaluation without the densecrf with the dice coefficient"""
    if context.net.n_classes > 1:
        avg_meters = {'loss': AverageMeter(),'miou': AverageMeter()}
        if miou_split:# show detail for iou of each class
            for item in context.datamaker.class_names:
                avg_meters["iou_{}".format(item)] = AverageMeter()
    else:
        avg_meters = {'loss': AverageMeter(),'iou': AverageMeter(),'pixel_error': AverageMeter(),'rand_error': AverageMeter(),'dice_coeff':AverageMeter()}

    context.net.eval()
    mask_type = torch.float32 if context.net.n_classes == 1 else torch.long
    batch_count = 0

    with tqdm(total=context.n_val, desc='Validation round') as pbar:
        show=False
        for batch  in context.val_loader:
            imgs = batch['image']
            true_masks = batch['mask']
            if 'weight' in batch:
                weight = batch['weight']
            else:
                weight = None
            imgs = imgs.to(device=context.device, dtype=torch.float32)
            if weight is not None:
                weight = weight.to(device=context.device, dtype=torch.float32)
            true_masks = true_masks.to(device=context.device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = context.net(imgs)

                if context.args.deep_supervision: #choose final
                    mask_pred = mask_pred[-1]

                if context.net.n_classes > 1:
                    loss = context.criterion(mask_pred, true_masks,epoch)
                    if not context.args.loss_reduce:
                        loss = loss.mean()
                    avg_meters['loss'].update(loss.cpu().item())
                    pbar.set_postfix(**{'val_loss': avg_meters['loss'].avg})
                    for i in range(imgs.shape[0]):
                        s_true_mask =  true_masks[i].cpu().detach().numpy()
                        s_pred = mask_pred[i].cpu().detach().numpy()
                        img = imgs[i].cpu().detach().numpy()
                        s_pred = np.argmax(s_pred,axis=0)
                        if not show:
                            context.datamaker.showrevert_cp2file(img,s_true_mask,s_pred,context.args.experiment,epoch)
                            show = True
                        miou,statisic= mIOU(s_pred,s_true_mask,context.net.n_classes)
                        avg_meters['miou'].update(miou)
                        if miou_split:
                            for key in statisic:
                                iou = (statisic[key]['tp']*1.0) / (statisic[key]['tp']+statisic[key]['fp']+statisic[key]['fn']+(-1e-5))
                                avg_meters["iou_{}".format(context.datamaker.class_names[key])].update(iou)
                else:
                    pred = torch.sigmoid(mask_pred)
                    pred_int = (pred > 0.5).int()
                    pred = (pred > 0.5).float()
                    if context.args.weight_loss:
                        loss = context.criterion(mask_pred, true_masks,epoch,weight)
                    else:
                        loss = context.criterion(mask_pred, true_masks,epoch)
                    avg_meters['loss'].update(loss.cpu().item())
                    # for i in range(imgs.shape[0]):
                    for i in range(imgs.shape[0]):
                        s_true_mask =  true_masks[i].cpu().detach().numpy()
                        s_pred = pred_int[i].cpu().detach().numpy()
                        if not show:# once for a epoch
                            if i==0:
                                vis.visualize_pred_to_file("./result/{}/epoch:{}_{}_{}.png".format(context.args.experiment,epoch,batch_count,i),imgs[i].cpu().detach().numpy(), s_true_mask, s_pred , title1="Original", title2="True", title3="Pred[0]")
                                stack = 'val_loss:{},iou:{},pixel_error:{},rand_error:{},dice_coeff:{}'.format(loss.cpu().item(),pixel_error(s_true_mask,s_pred),IOU(s_true_mask,s_pred),rand_error(s_true_mask,s_pred),dice_coeff(pred, true_masks).item())
                                with open("./result/{}/epoch:{}_{}_{}.txt".format(context.args.experiment,epoch,batch_count,i),'w') as f:    #设置文件对象
                                    f.write(stack)
                            show = True
                        avg_meters['pixel_error'].update(pixel_error(s_true_mask,s_pred))
                        avg_meters['iou'].update(IOU(s_true_mask,s_pred))
                        avg_meters['rand_error'].update(rand_error(s_true_mask,s_pred))
                        avg_meters['dice_coeff'].update(dice_coeff(pred, true_masks).item())
                        pbar.set_postfix(**{'val_loss': avg_meters['loss'].avg,'iou': avg_meters['iou'].avg,'pixel_error': avg_meters['pixel_error'].avg,'rand_error': avg_meters['rand_error'].avg})
            pbar.update(imgs.shape[0])
            batch_count+=1
        pbar.close()

    context.net.train()
    ret = OrderedDict()
    if context.net.n_classes >1:
        ret['loss'] = avg_meters['loss'].avg
        ret['mIOU'] = avg_meters['miou'].avg
        if miou_split:# show detail for iou of each class
            for item in context.datamaker.class_names:
                ret["iou_{}".format(item)] = avg_meters["iou_{}".format(item)].avg
    else:
        ret['loss'] = avg_meters['loss'].avg
        ret['iou'] = avg_meters['iou'].avg
        ret['pixel_error'] = avg_meters['pixel_error'].avg
        ret['rand_error'] = avg_meters['rand_error'].avg
        ret['dice_coeff'] = avg_meters['dice_coeff'].avg
    return ret

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
        criterion = losses.__dict__[args.loss](args)
        if args.device=="cuda":
            criterion.cuda()

    return criterion

#main entry
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    context.args = get_args()
    if context.args.device == 'cuda':
        context.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(context.args.device_id)
    else:
        context.device = torch.device('cpu')
    logging.info(f'Using device {context.device}')
    checkpoint = None
    load_from = None
    if context.args.load:
        load_from = context.args.load #备份加载位置，因为args会被替换
        checkpoint = torch.load(context.args.load, map_location=context.device)
        if 'args' in checkpoint:#兼容设置：因为旧版的部分运行保存没有保存参数args，所以有些读取是没有这个参数的 以免报错
            old_args = context.args
            context.args = checkpoint['args']
            if len(old_args.no_replace) != 0:
                keeps = old_args.no_replace.split(',')
                for keep in keeps:
                    context.args.__dict__[keep] = old_args.__dict__[keep]     
            logging.info(f'''reload training:
            args:          {context.args}
            ''')
    #set seed
    set_seed(context.args.seed)
    # init context
    context.net = archs.__dict__[context.args.arch](context.args)
    context.optimizer = get_optimizer(context.args,context.net)
    context.scheduler = get_scheduler(context.args,context.optimizer)
    context.criterion = get_criterion(context.args,context.net)
    start_epoch =0
    trainer = trainers.__dict__[context.args.trainer]()


    #keep initial net weight norm info here
    weight_norms = weight_norm_init(context.net,context.args)
    init_norms = OrderedDict()
    for idx in range(context.args.num_classes):
        init_norms['final_norm_{}'.format(idx)] = weight_norms[idx]

    #mkdirs for each experiment
    try:
        os.mkdir('./result/{}'.format(context.args.experiment))
    except OSError:
        pass
    print(context.net)
    logging.info(f'Network:\n'
                 f'\t{context.args.input_channels} input channels\n'
                 f'\t{context.args.num_classes} output channels (classes)\n')
    if load_from is not None:
        if 'args' in checkpoint:#新版保存了 网络状态，优化器状态等，旧版没有，作兼容
            context.net.load_state_dict(checkpoint['net'])
            if context.args.continnue:
                context.optimizer.load_state_dict(checkpoint['optimizer'])
                for state in context.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                start_epoch = checkpoint['epoch']
            logging.info(f'Model loaded from {load_from} in epoch {start_epoch}')
        else:
            context.net.load_state_dict(checkpoint)

    context.net.to(device=context.device)

    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # init data set here:
    #global datamaker
    context.datamaker = datasets.__dict__[context.args.dataset](context.args)
    context.train_loader,context.val_loader,context.n_train,context.n_val =context.datamaker(context.args)
    logging.info(f'''Starting training:
        args:          {context.args}
    ''')

    #for csv log
    log = OrderedDict()
    writer = SummaryWriter(comment=f'_ex.{context.args.experiment}_{context.args.optimizer}_LR_{context.args.lr}_BS_{context.args.batchsize}_model_{context.args.arch}')
    savepoint=savepoints.__dict__[context.args.savepoint](context.args)
    try:
        rounds = range(start_epoch,context.args.epochs)
        for epoch in rounds:
            # train
            train_log = trainer(context =context,epoch=epoch)
            #validate
            val_log = eval_net(context = context,epoch=epoch)

            if context.scheduler is not None and context.args.scheduler == 'ReduceLROnPlateau':
                context.scheduler.step(val_log['loss'])
            else:
                context.scheduler.step()

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

            for tag, value in context.net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), (epoch+1))
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), (epoch+1))
            logging.info('==================================================>')
            if context.net.n_classes > 1:
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
            writer.add_scalar('learning_rate', context.optimizer.param_groups[0]['lr'], (epoch+1))

            #force to save check point at last epoch
            if context.args.force_save_last and epoch == context.args.epochs-1:
                state = {'args':context.args,'net':context.net.state_dict(), 'optimizer':context.optimizer.state_dict(), 'epoch':epoch}
                torch.save(state,context.args.dir_checkpoint + f'CP_ex.{context.args.experiment}_epoch{epoch + 1}_{context.args.arch}_{context.args.dataset}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
            # save check point by condition
            if context.args.save_check_point:
                if context.args.save_mode == 'by_best':
                    if savepoint.is_new_best(val_log=val_log):
                        try:
                            os.mkdir(context.args.dir_checkpoint)
                            logging.info('Created checkpoint directory')
                        except OSError:
                            pass
                        state = {'args':context.args,'net':context.net.state_dict(), 'optimizer':context.optimizer.state_dict(), 'epoch':epoch}
                        torch.save(state,context.args.dir_checkpoint + f'CP_ex.{context.args.experiment}_epoch{epoch + 1}_{context.args.arch}_{context.args.dataset}.pth')
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
                        os.mkdir(context.args.dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    state = {'args':context.args,'net':context.net.state_dict(), 'optimizer':context.optimizer.state_dict(), 'epoch':epoch}
                    torch.save(state,context.args.dir_checkpoint + f'CP_ex.{context.args.experiment}_epoch{epoch + 1}_{context.args.arch}_{context.args.dataset}.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')

        # saving initial weight norm info to final log files
        init_log = OrderedDict()
        for m_key in init_norms:
            #for single file in csv
            if '{}_{}'.format('train',m_key) not in init_log:
                init_log['{}_{}'.format('train',m_key)]=[]
            init_log['{}_{}'.format('train',m_key)].append(init_norms[m_key])
        #for csv
        pd.DataFrame(init_log).to_csv('./result/{}_init.csv'.format(context.args.experiment),index=None)
        pd.DataFrame(log).to_csv('./result/{}.csv'.format(context.args.experiment),index=None)
        writer.close()
    except KeyboardInterrupt:
        if context.args.save_check_point:
            state = {'args':context.args,'net':context.net.state_dict(), 'optimizer':context.optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, context.args.dir_checkpoint+f'INTERRUPTED_ex.{context.args.experiment}_epoch{epoch + 1}_{context.args.arch}_{context.args.dataset}.pth')
            logging.info('Saved interrupt in {} epoch'.format(epoch))
        writer.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
