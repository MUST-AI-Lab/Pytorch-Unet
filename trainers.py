import torch
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import sum_tensor,softmax_helper
from utils.tools import AverageMeter,str2bool,softmax_helper
from utils.weights_collate import default_collate_with_weight,label2_baseline_weight_by_prior,label2distribute,distribution2tensor,label_count

import numpy as np
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
from utils.tools import weight_norm,weight_norm_init
from tqdm import tqdm

__all__ = ["STDTrainer","CBLossTrainer"]


#trainer for cbloss
#need hyper paramemter beta
class CBLossTrainer:
    def __init__(self):
        self.ajust_loss =-1
        pass

    def __call__(self,context,epoch):
        context.net.train()
        avg_meters = {'loss': AverageMeter()}
        if not context.args.loss_reduce:
            for idx in range(context.args.num_classes):
                avg_meters['loss_{}'.format(idx)] = AverageMeter()

        for idx in range(context.args.num_classes):# for init norm statistic
            avg_meters['final_norm_{}'.format(idx)] = AverageMeter()
            avg_meters['loss_gd_norm_{}'.format(idx)] = AverageMeter()

        samples_per_cls = None
        with tqdm(total=context.n_train, desc=f'Epoch {epoch + 1}/{context.args.epochs}', unit='img') as pbar:
            iter =0
            for batch in context.train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                if 'weight' in batch:
                    weight = batch['weight']
                    count = weight
                    if samples_per_cls is None:
                        samples_per_cls = np.squeeze(count.numpy()+1,axis=0)
                    else:
                        samples_per_cls = samples_per_cls + np.squeeze(count.numpy(),axis=0)
                    beta = context.args.beta
                    effective_num = 1.0 - np.power(beta, samples_per_cls)
                    weights = (1.0 - beta) / effective_num
                    #区分 weights and weight
                    weight = distribution2tensor(context.args.num_classes,weights,true_masks.numpy())
                    weight = weight * np.sum(count.numpy())
                    weight = torch.tensor(weight).float()
                    

                else:
                    weight = None
                assert imgs.shape[1] == context.net.n_channels, \
                        f'Network has been defined with {context.net.n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                imgs = imgs.to(device=context.device, dtype=torch.float32)
                if weight is not None:
                    weight = weight.to(device=context.device, dtype=torch.float32)
                mask_type = torch.float32 if context.net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=context.device, dtype=mask_type)

                masks_pred = context.net(imgs)
                # compute output
                if context.args.deep_supervision:
                    loss = 0
                    for output in masks_pred:
                        if weight is not None and context.args.weight_loss:
                            loss += context.criterion(output, true_masks,epoch,weight)
                        else:
                            loss += context.criterion(output, true_masks,epoch)
                    loss /= len(masks_pred)
                else:
                    if weight is not None and context.args.weight_loss:
                        loss = context.criterion(masks_pred, true_masks,epoch,weight)
                    else:
                        loss = context.criterion(masks_pred, true_masks,epoch)

                #print(loss)
                if not context.args.loss_reduce:
                    avg_meters['loss'].update(loss.mean().cpu().item())
                    loss_np =  np.array([t.cpu().detach().numpy() for t in loss])
                    true_masks_np = np.array([t.cpu().detach().numpy() for t in true_masks] )
                    for idx in range(context.args.num_classes):
                        tmp_loss = np.sum(loss_np*(true_masks_np==idx).astype(np.int))
                        tmp_count = np.sum((true_masks_np==idx).astype(np.int))
                        avg_meters['loss_{}'.format(idx)].update(tmp_loss/(tmp_count+1))#no zero div
                    loss = loss.mean()
                else:
                    avg_meters['loss'].update(loss.cpu().item())
                    if self.ajust_loss <0 :
                        self.ajust_loss = 3.5/loss.cpu().item()

                pbar.set_postfix(**{'train_loss': avg_meters['loss'].avg})
                iter +=1
                if context.args.accumulation_step==1:
                    context.optimizer.zero_grad()
                    loss.backward()
                    #----------------------------
                    weight_norms,loss_norms = weight_norm(context.net,context.args) #collect weight norm for final layer
                    for i in range(context.args.num_classes):
                        avg_meters['final_norm_{}'.format(i)].update(weight_norms[i])
                        avg_meters['loss_gd_norm_{}'.format(i)].update(loss_norms[i])
                    #----------------------------
                    nn.utils.clip_grad_value_(context.net.parameters(), 0.1)
                    context.optimizer.step()
                else:
                    loss = loss/context.args.accumulation_step
                    loss.backward()

                    if(iter%context.args.accumulation_step)==0:
                        # optimizer the net
                        context.optimizer.step()        # update parameters of net
                        context.optimizer.zero_grad()   # reset gradient

                pbar.update(imgs.shape[0])
                if iter >1000000:#这些代码是测试用的 可以删除掉
                    break
            pbar.close()
            redict = None
            if not context.args.loss_reduce:
                redict = OrderedDict([('loss', avg_meters['loss'].avg)])
                for idx in range(context.args.num_classes):
                    redict['loss_{}'.format(idx)] = avg_meters['loss_{}'.format(idx)].avg
                    redict['final_norm_{}'.format(idx)] = avg_meters['final_norm_{}'.format(idx)].avg
                    redict['loss_gd_norm_{}'.format(idx)] = avg_meters['loss_gd_norm_{}'.format(idx)].avg
            else:
                redict = OrderedDict([('loss', avg_meters['loss'].avg)])
                for idx in range(context.args.num_classes):
                    redict['final_norm_{}'.format(idx)] = avg_meters['final_norm_{}'.format(idx)].avg
                    redict['loss_gd_norm_{}'.format(idx)] = avg_meters['loss_gd_norm_{}'.format(idx)].avg
        return redict


class STDTrainer:
    def __init__(self):
        pass

    def __call__(self,context,epoch):
        context.net.train()
        avg_meters = {'loss': AverageMeter()}
        if not context.args.loss_reduce:
            for idx in range(context.args.num_classes):
                avg_meters['loss_{}'.format(idx)] = AverageMeter()

        for idx in range(context.args.num_classes):# for init norm statistic
            avg_meters['final_norm_{}'.format(idx)] = AverageMeter()
            avg_meters['loss_gd_norm_{}'.format(idx)] = AverageMeter()

        with tqdm(total=context.n_train, desc=f'Epoch {epoch + 1}/{context.args.epochs}', unit='img') as pbar:
            iter =0
            for batch in context.train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                if 'weight' in batch:
                    weight = batch['weight']
                else:
                    weight = None
                assert imgs.shape[1] == context.net.n_channels, \
                    f'Network has been defined with {context.net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=context.device, dtype=torch.float32)
                if weight is not None:
                    weight = weight.to(device=context.device, dtype=torch.float32)
                mask_type = torch.float32 if context.net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=context.device, dtype=mask_type)

                masks_pred = context.net(imgs)
                # compute output
                if context.args.deep_supervision:
                    loss = 0
                    for output in masks_pred:
                        if weight is not None and context.args.weight_loss:
                            loss += context.criterion(output, true_masks,epoch,weight)
                        else:
                            loss += context.criterion(output, true_masks,epoch)
                    loss /= len(masks_pred)
                else:
                    if weight is not None and context.args.weight_loss:
                        loss = context.criterion(masks_pred, true_masks,epoch,weight)
                    else:
                        loss = context.criterion(masks_pred, true_masks,epoch)

                #print(loss)
                if not context.args.loss_reduce:
                    avg_meters['loss'].update(loss.mean().cpu().item())
                    loss_np =  np.array([t.cpu().detach().numpy() for t in loss])
                    true_masks_np = np.array([t.cpu().detach().numpy() for t in true_masks] )
                    for idx in range(context.args.num_classes):
                        tmp_loss = np.sum(loss_np*(true_masks_np==idx).astype(np.int))
                        tmp_count = np.sum((true_masks_np==idx).astype(np.int))
                        avg_meters['loss_{}'.format(idx)].update(tmp_loss/(tmp_count+1))#no zero div
                    loss = loss.mean()
                else:
                    avg_meters['loss'].update(loss.cpu().item())

                pbar.set_postfix(**{'train_loss': avg_meters['loss'].avg})
                iter +=1
                if context.args.accumulation_step==1:
                    context.optimizer.zero_grad()
                    loss.backward()
                    #----------------------------
                    weight_norms,loss_norms = weight_norm(context.net,context.args) #collect weight norm for final layer
                    for i in range(context.args.num_classes):
                        avg_meters['final_norm_{}'.format(i)].update(weight_norms[i])
                        avg_meters['loss_gd_norm_{}'.format(i)].update(loss_norms[i])
                    #----------------------------
                    nn.utils.clip_grad_value_(context.net.parameters(), 0.1)
                    context.optimizer.step()
                else:
                    loss = loss/context.args.accumulation_step
                    loss.backward()

                    if(iter%self.context.args.accumulation_step)==0:
                        # optimizer the net
                        context.optimizer.step()        # update parameters of net
                        context.optimizer.zero_grad()   # reset gradient

                pbar.update(imgs.shape[0])
                if iter >1000000:#这些代码是测试用的 可以删除掉
                    break
            pbar.close()
            redict = None
            if not context.args.loss_reduce:
                redict = OrderedDict([('loss', avg_meters['loss'].avg)])
                for idx in range(context.args.num_classes):
                    redict['loss_{}'.format(idx)] = avg_meters['loss_{}'.format(idx)].avg
                    redict['final_norm_{}'.format(idx)] = avg_meters['final_norm_{}'.format(idx)].avg
                    redict['loss_gd_norm_{}'.format(idx)] = avg_meters['loss_gd_norm_{}'.format(idx)].avg
            else:
                redict = OrderedDict([('loss', avg_meters['loss'].avg)])
                for idx in range(context.args.num_classes):
                    redict['final_norm_{}'.format(idx)] = avg_meters['final_norm_{}'.format(idx)].avg
                    redict['loss_gd_norm_{}'.format(idx)] = avg_meters['loss_gd_norm_{}'.format(idx)].avg
        return redict

