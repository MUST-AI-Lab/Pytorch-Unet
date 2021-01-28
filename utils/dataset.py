from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import utils.mic.dataset_helper as helper
import cv2
import os
import numpy as np
import scipy.misc as m
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.tools import mask2onehot,onehot2mask,sample_array,to_int
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from utils.weights_collate import default_collate_with_weight,label2_baseline_weight_by_prior,label2distribute,distribution2tensor,label_count
import pandas as pd
from utils.mic.dataset_helper import load_from_single_page_tiff,load_from_multi_page_tiff

__all__ = ['HeLa','U373','Cam2007DatasetV2','KeyBoard','KeyBoard2','ISBI2012','DSB','CityScape']


class SegDataSet_T(Dataset):
    def __init__(self,args):
        self.data_dir =args.data_dir
        self.args = args
        self.cmap = self.labelcolormap(args.num_classes)
        self.class_names = self.get_classes_names()

    def labelcolormap(self,N):
        raise NotImplementedError("labelcolormap: not implemented!")

    def get_classes_names(self):
        raise NotImplementedError("get_classes_names: not implemented!")

    #get pair from dataset
    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        raise NotImplementedError("get_pairs: not implemented!")

    #get current dataset statistic
    def get_disturibution(self):
        keeper = dict()
        keeper['id'] = []
        for item in self.class_names:
            keeper[item] = []
        for _,label,ids in tqdm(self.pairs):
            keeper['id'].append(ids)
            total = np.prod(label.shape)
            total_pixel =0
            for i in range(len(self.class_names)):
                state = (label==i).astype(np.int)
                total_pixel +=  np.sum(state)
                keeper[self.class_names[i]].append((np.sum(state))/total)
            #check sum
            assert total == total_pixel,"not total pixel"

        keeper['id'].append('total')
        summary_factor = []
        for item in self.class_names:#total
            factor = np.sum(keeper[item])/len(self.pairs)
            keeper[item].append(factor)
            summary_factor.append(factor)
        print(summary_factor)
        summary_factor2 = summary_factor.copy()
        idx = sorted(range(len(summary_factor2)), key=lambda k: summary_factor2[k],reverse=True)
        print(idx)
        self.summary_factor = summary_factor
        
    def __len__(self):
        return len(self.pairs)
        # map rbg to label
    
    def rgb2label(self,rgb):
        for index in range(len(self.cmap)):
            u = rgb==self.cmap[index]
            if u[0] and u[1] and u[2]:
                return index
        return 0
    
    # change label from rbg to class
    def handle_label(self,label):
        label_axes = (label.shape[0],label.shape[1])
        new_label = np.zeros(label_axes).astype(np.uint8)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                new_label[i][j] = self.rgb2label(label[i][j])
        return new_label

    def label2rgb(self,label):
        return self.cmap[label]

    def revert_label2rgb(self, label):
        label_axes = (label.shape[0],label.shape[1],3)
        new_label = np.zeros(label_axes).astype(np.uint8)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                color = self.label2rgb(label[i,j])
                new_label[i,j,:]=color
        return new_label

    def showrevert_cp2file(self,image,label,pred,experiment,epoch=0):
        gray_img = False
        rev_label = self.revert_label2rgb(label)
        rev_pred = self.revert_label2rgb(pred)
        if image.max()<1.1:
            image *= 255
        if image.min()<0:    
            image+=128
        if image.shape[0] == 3:
            image=image.transpose(1, 2, 0).astype('float32')
        if image.shape[0] == 1:
            image=np. squeeze(image,axis=0)    
            gray_img=True
        if not gray_img:
            image = Image.fromarray(image.astype(np.uint8))
        else:
            image=Image.fromarray(image.astype(np.uint8),mode='L')
        rev_label = Image.fromarray(np.uint8(rev_label))
        rev_pred= Image.fromarray(np.uint8(rev_pred))
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        if gray_img:
            axes[0].imshow(image,cmap ='gray')
        else:
            axes[0].imshow(image)
        axes[1].imshow(rev_label)
        axes[2].imshow(rev_pred)
        plt.savefig("{}/{}/{}.png".format('result',experiment,epoch))

    def showrevert_cp2file_origin(self,image,label,pred,experiment,epoch=0):
        gray_img=False
        rev_label = self.revert_label2rgb(label)
        rev_pred = self.revert_label2rgb(pred)
        rev_pred_out = rev_pred
        if image.max()<1.1:
            image *= 255
        if image.min()<0:    
            image+=128
        if image.shape[0] == 3:
            image=image.transpose(1, 2, 0).astype('float32')
        if image.shape[0] == 1:
            image=np. squeeze(image,axis=0)
            gray_img=True
        if not gray_img:
            image = Image.fromarray(image.astype(np.uint8))
        else:
            image=Image.fromarray(image.astype(np.uint8),mode='L')
        
        rev_label = Image.fromarray(np.uint8(rev_label))
        rev_pred= Image.fromarray(np.uint8(rev_pred))
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        if gray_img:
            axes[0].imshow(image,cmap ='gray')
        else:
            axes[0].imshow(image)
        axes[1].imshow(rev_label)
        axes[2].imshow(rev_pred)
        plt.savefig("{}/{}/cp_{}.png".format('result',experiment,epoch))
        cv2.imwrite("{}/{}/{}.png".format('result',experiment,epoch),img=cv2.cvtColor(rev_pred_out.astype(np.uint8),cv2.COLOR_RGB2BGR))

    def __getitem__(self, idx):
        image,label,id = self.pairs[idx]
        #pre_process
        img = image.astype('float32')/ 255

        #get weight process
        if len(img.shape)==3: #switch channel to first
            img = image.transpose(2, 0, 1).astype('float32')
        else:# gray channel is 1
            img = np.expand_dims(image,axis=0)
        if self.args.weight_type == 'global_test_weight':
            weight=np.zeros_like(label).astype(np.float)
            max_di = np.max(self.summary_factor)
            e = 2.7182
            for i in range(len(self.class_names)):
                pt = self.summary_factor[i]/max_di
                weight[label == i] = 1.5*(1/e**pt)
        elif self.args.weight_type == 'single_test_weight':
            weight = self.label2pixel_one_image(label)
        elif self.args.weight_type == 'global_test_distrubution':
            e = 2.7182
            weight=np.ones_like(self.summary_factor).astype(np.float)
            weight=weight*self.summary_factor
            max_di = np.max(self.summary_factor)
            weight = weight/max_di
            weight = 1.5 *(1/e**weight)
        elif self.args.weight_type == 'global_distrubution':
            weight=np.ones_like(self.summary_factor).astype(np.float)
            weight=weight*self.summary_factor
        elif self.args.weight_type == 'single_distrubution':
            weight=summary_factor = label2distribute(len(self.class_names),label)
        elif self.args.weight_type == 'batch_distrubution':
            #注意，计算一个batch的统计量权重，需要collate函数配合，并不是在这里计算的。
            # 故这个选项下是特殊的返回值
            distribution = self.label2distribute(label)
            return  {
                    'image': img,
                    'mask': label,
                    'batch_distrubution':distribution,
                    'class_nums':len(self.class_names)
            }
        elif self.args.weight_type == 'single_baseline_weight':
            summary_factor = label2distribute(len(self.class_names),label)
            weight = label2_baseline_weight_by_prior(len(self.class_names),summary_factor,label)
        elif self.args.weight_type == 'global_baseline_weight':
            weight = label2_baseline_weight_by_prior(len(self.class_names),self.summary_factor,label)
            #weight = self.label2weight_global_prior(label)
        elif self.args.weight_type == 'batch_baseline_weight':
            #注意，计算一个batch的统计量权重，需要collate函数配合，并不是在这里计算的。
            # 故这个选项下是特殊的返回值
            # from utils.weights_collate import default_collate_with_weight
            distribution = self.label2distribute(label)
            return  {
                    'image': img,
                    'mask': label,
                    'batch_baseline_weight':distribution,
                    'class_nums':len(self.class_names)
            }
        elif self.args.weight_type == 'batch_test_weight':
            #注意，计算一个batch的统计量权重，需要collate函数配合，并不是在这里计算的。
            # 故这个选项下是特殊的返回值
            # from utils.weights_collate import default_collate_with_weight
            distribution = self.label2distribute(label)
            return  {
                    'image': img,
                    'mask': label,
                    'batch_test_weight':distribution,
                    'class_nums':len(self.class_names)
            }
        elif self.args.weight_type == 'single_distribute_weight':
            summary_factor = label2distribute(len(self.class_names),label)
            weight = distribution2tensor(len(self.class_names),summary_factor,label)
        elif self.args.weight_type == 'global_distribute_weight':
            weight = distribution2tensor(len(self.class_names),self.summary_factor,label)
        elif self.args.weight_type == 'batch_distribute_weight':
            #注意，计算一个batch的统计量权重，需要collate函数配合，并不是在这里计算的。
            # 故这个选项下是特殊的返回值
            # from utils.weights_collate import default_collate_with_weight
            distribution = self.label2distribute(label)
            return  {
                    'image': img,
                    'mask': label,
                    'batch_distribute_weight':distribution,
                    'class_nums':len(self.class_names)
            }
        elif self.args.weight_type == 'single_count':
            weight = label_count(len(self.class_names),label)
        elif self.args.weight_type == 'none':
            #防止两个命令冲突
            return {
                    'image': torch.from_numpy(img).type(torch.FloatTensor),
                    'mask': torch.from_numpy(label).type(torch.IntTensor)
            }
        else:
            assert None ,"uknow weight type"

        if self.args.weight_loss:
            return {
                    'image': torch.from_numpy(img).type(torch.FloatTensor),
                    'mask': torch.from_numpy(label).type(torch.IntTensor),
                    'weight':torch.from_numpy(weight).type(torch.FloatTensor)
            }
        else:
            return {
                    'image': torch.from_numpy(img).type(torch.FloatTensor),
                    'mask': torch.from_numpy(label).type(torch.IntTensor)
            }

    # in preprocess, save img to files
    def pairs2files(self,out_path,img="IMG",GT="GT"):
        i =0
        for cityscape,label in tqdm(self.pairs):
            np.save('{}/{}/{}.npy'.format(out_path,img,i ), cityscape)
            np.save('{}/{}/{}.npy'.format(out_path,GT,i ), label)
            i +=1

    # a pop up weight for each image stastic
    def label2pixel_one_image(self,label):
        weight = np.zeros_like(label, dtype='float32')
        total = np.prod(label.shape)
        factor = np.zeros(len(self.class_names), dtype='float32')
        for i in range(len(self.class_names)):
            state = (label==i).astype(np.int)
            factor[i] = np.sum(state)/total
        max_di = np.max(factor)
        e = 2.7182
        for i in range(len(self.class_names)):
            pt = factor[i]/max_di
            weight[label == i] = 1.5*(1/e**pt)
        return weight

    # baseline weight for global stastic
    def label2weight_global_prior(self,label, w_min: float = 1., w_max: float = 2e5):
        weight = np.zeros_like(label, dtype='float32')
        K = len(self.class_names) - 1
        for i in range(len(self.class_names)):
            weight[label == i] = 1 / (K + 1) * (1/(self.summary_factor[i]+2e-5))#modify to no zero divide
        # we add clip for learning stability
        # e.g. if we catch only 1 voxel of some component, the corresponding weight will be extremely high (~1e6)
        return np.clip(weight, w_min, w_max)

    # baseline weight for each image stastic
    def label2weight(self,label, w_min: float = 1., w_max: float = 2e5):
        weight = np.zeros_like(label, dtype='float32')
        K = len(self.class_names) - 1
        N = np.prod(label.shape)
        for i in range(len(self.class_names)):
            weight[label == i] = N+1 / ((K + 1) * np.sum(label == i)+1)#modify to no zero divide
        # we add clip for learning stability
        # e.g. if we catch only 1 voxel of some component, the corresponding weight will be extremely high (~1e6)
        return np.clip(weight, w_min, w_max)

    # baseline distribution for one
    def label2distribute(self,label, w_min: float = 1., w_max: float = 2e5):
        weight = np.ones(len(self.class_names), dtype='float32')
        N = np.prod(label.shape)
        for i in range(len(self.class_names)):
            weight[i] =(np.sum(label == i)) / N #Make sure here should be same denominator, Otherwise the value is not allowed to be used to get the weight
        return weight

    # get pixel distribution from data set
    def statistic_dataset(self,count=2):
        keeper = dict()
        keeper['id'] = []
        for item in self.class_names:
            keeper[item] = []
        for _,label,ids in tqdm(self.pairs):
            keeper['id'].append(ids)
            total = np.prod(label.shape)
            total_pixel =0
            for i in range(self.args.num_classes):
                state = (label==i).astype(np.int)
                total_pixel +=  np.sum(state)
                keeper[self.class_names[i]].append((np.sum(state))/total)
            #check sum 
            assert total == total_pixel,"not total pixel"

        keeper['id'].append('total')
        summary_factor = []
        for item in self.class_names:#total
            factor = np.sum(keeper[item])/len(self.pairs)
            keeper[item].append(factor)
            summary_factor.append(factor)


        fig, ax = plt.subplots()
        summary_factor = np.array(summary_factor )
        idx = np.argsort(summary_factor)
        summary_factor = np.sort(summary_factor)
        names = [self.class_names[k] for  k in idx]
        colors = [self.cmap[k] for k in idx]
        for i in range(len(self.class_names)):#total
            ax.bar(names[i], summary_factor[i],color=self.RGB_to_Hex("{},{},{}".format(colors[i][0],colors[i][1],colors[i][2])))
        for a,b in zip(names,summary_factor):
            plt.text(a, b+0, '%.{}f'.format(count) % b, ha='center', va= 'bottom',fontsize=10)
        plt.xticks(rotation = 270,fontsize=10)
        plt.title('Distribution of pixels count')
        plt.show()

        dataframe = pd.DataFrame.from_dict(keeper)
        dataframe.to_csv("{}_{}.csv".format(self.args.experiment,self.__len__()),index=False)
    
    def RGB_to_Hex(self,tmp):
        rgb = tmp.split(',')#将RGB格式划分开来
        strs = '#'
        for i in rgb:
            num = int(i)#将str转int
            #将R、G、B分别转化为16进制拼接转换并大写
            strs += str(hex(num))[-2:].replace('x','0').upper()
        return strs
    
    def uint82bin(self, n, count=8):
        return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

class KeyBoard2(SegDataSet_T):
    def __init__(self,args):
        SegDataSet_T.__init__(self,args)
    
    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [0,0,0]
        cmap[1] = [128,0,0]
        cmap[2] = [0,128,0]
        cmap[3] = [128,128,0]
        return cmap

    def get_classes_names(self):
        return ["Background", "Key","Key_Light","Leak"]

    def __call__(self,args):
        train = KeyBoard2(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = KeyBoard2(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        if True:
            train.statistic_dataset()
            val.statistic_dataset()
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    #get pair from png
    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = sample_array("{}/{}".format(self.data_dir,cfg))
        print("There are {} pictures.".format(len(self.data_fns)))
        print("files name arrays are {}".format(self.data_fns))
        for ids in tqdm(self.data_fns):
            #image = np.load('{}/{}/{}.npy'.format(self.data_dir,img,ids ))
            #label = np.load('{}/{}/{}.npy'.format(self.data_dir,GT,ids ))
            image = Image.open('{}/{}/{}{}'.format(self.data_dir,img,ids,self.args.img_ext)).convert("RGB")
            image = np.array(image)
            label =  Image.open('{}/{}/{}{}'.format(self.data_dir,GT,ids,self.args.img_ext)).convert("RGB")
            label = np.array(label)
            label = self.handle_label(label)
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))
      
class KeyBoard(SegDataSet_T):
    def __init__(self,args):
        SegDataSet_T.__init__(self,args)
        
    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [0,0,0]
        cmap[1] = [128,0,0]
        return cmap

    def get_classes_names(self):
        return ["Background", "Leak"]
    
    def __call__(self,args):
        train = KeyBoard(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = KeyBoard(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    #get pair from png
    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = sample_array("{}/{}".format(self.data_dir,cfg))
        print("There are {} pictures.".format(len(self.data_fns)))
        print("files name arrays are {}".format(self.data_fns))
        for ids in tqdm(self.data_fns):
            #image = np.load('{}/{}/{}.npy'.format(self.data_dir,img,ids ))
            #label = np.load('{}/{}/{}.npy'.format(self.data_dir,GT,ids ))
            image = Image.open('{}/{}/{}{}'.format(self.data_dir,img,ids,self.args.img_ext)).convert("RGB")
            image = np.array(image)
            label =  Image.open('{}/{}/{}{}'.format(self.data_dir,GT,ids,self.args.img_ext)).convert("RGB")
            label = np.array(label)
            label = self.handle_label(label)
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))
    
class Cam2007DatasetV2(SegDataSet_T):
    def __init__(self,args):
        SegDataSet_T.__init__(self,args)
         
    def get_classes_names(self):
        return [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]
    
    def __call__(self,args):
        train = Cam2007DatasetV2(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = Cam2007DatasetV2(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        if False:
            train.statistic_dataset()
            val.statistic_dataset()
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [64,128,64]
        cmap[1] = [192,0,128]
        cmap[2] = [0,128,192]
        cmap[3] = [0,128,64]
        cmap[4] = [128,0,0]
        cmap[5] = [64,0,128]
        cmap[6] = [64,0,192]
        cmap[7] = [192,128,64]
        cmap[8] = [192,192,128]
        cmap[9] = [64,64,128]
        cmap[10] = [128,0,192]
        cmap[11] = [192,0,64]
        cmap[12] = [128,128,64]
        cmap[13] = [192,0,192]
        cmap[14] = [128,64,64]
        cmap[15] = [64,192,128]
        cmap[16] = [64,64,0]
        cmap[17] = [128,64,128]
        cmap[18] = [128,128,192]
        cmap[19] = [0,0,192]
        cmap[20] = [192,128,128]
        cmap[21] = [128,128,128]
        cmap[22] = [64,128,192]
        cmap[23] = [0,0,64]
        cmap[24] = [0,64,64]
        cmap[25] = [192,64,128]
        cmap[26] = [128,128,0]
        cmap[27] = [192,128,192]
        cmap[28] = [64,0,64]
        cmap[29] = [192,192,0]
        cmap[30] = [0,0,0]
        cmap[31] = [64,192,0]
        return cmap
    
    #get pair from npy
    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = sample_array("{}/{}".format(self.data_dir,cfg))
        print("There are {} pictures.".format(len(self.data_fns)))
        print("files name arrays are {}".format(self.data_fns))
        for ids in self.data_fns:
            image = np.load('{}/{}/{}.npy'.format(self.data_dir,img,ids ))
            label = np.load('{}/{}/{}.npy'.format(self.data_dir,GT,ids ))
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))

class HeLa(SegDataSet_T):
    def __init__(self,args):
        assert args.target_set is not None,"target_set should not None"
        assert args.data_dir is not None,"data_path should not None"
        SegDataSet_T.__init__(self,args)
        self.target_set = args.target_set
        self.scale = args.scale
        self.vis = False
        helper.DATA_PATH=args.data_dir
    
    def get_classes_names(self):
        return ["Background", "Cell"]

    def __call__(self,args):
        train = HeLa(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = HeLa(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [0,0,0]
        cmap[1] = [255,255,255]
        return cmap

    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        instance=False 
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = sample_array("{}/{}".format(self.data_dir,cfg))
        print("There are {} pictures.".format(len(self.data_fns)))
        print("files name arrays are {}".format(self.data_fns))
        for ids in self.data_fns:
            image = load_from_single_page_tiff('{}/{}/t{}.tif'.format(self.data_dir,img,ids ))
            label = load_from_single_page_tiff('{}/{}/man_seg{}.tif'.format(self.data_dir,GT,ids ))
            if not instance:
                label = (label > 0).astype(np.int)
            mask = np.unique(label)
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))

class U373(SegDataSet_T):
    def __init__(self,args):
        assert args.target_set is not None,"target_set should not None"
        assert args.data_dir is not None,"data_path should not None"
        SegDataSet_T.__init__(self,args)
        self.target_set = args.target_set
        self.scale = args.scale
        self.vis = False
        helper.DATA_PATH=args.data_dir
    
    def get_classes_names(self):
        return ["Background", "Cell"]

    def __call__(self,args):
        train = U373(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = U373(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [0,0,0]
        cmap[1] = [255,255,255]
        return cmap

    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        instance=False 
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = sample_array("{}/{}".format(self.data_dir,cfg))
        print("There are {} pictures.".format(len(self.data_fns)))
        print("files name arrays are {}".format(self.data_fns))
        for ids in self.data_fns:
            image = load_from_single_page_tiff('{}/{}/t{}.tif'.format(self.data_dir,img,ids ))
            label = load_from_single_page_tiff('{}/{}/man_seg{}.tif'.format(self.data_dir,GT,ids ))
            if not instance:
                label = (label > 0).astype(np.int)
            mask = np.unique(label)
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))

class ISBI2012(SegDataSet_T):
    def __init__(self,args):
        assert args.target_set is not None,"target_set should not None"
        assert args.data_dir is not None,"data_path should not None"
        SegDataSet_T.__init__(self,args)
        self.target_set = args.target_set
        self.scale = args.scale
        self.vis = False
        helper.DATA_PATH=args.data_dir
    
    def get_classes_names(self):
        return ["Cell_Board", "Cell"]

    def __call__(self,args):
        train = ISBI2012(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = ISBI2012(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [0,0,0]
        cmap[1] = [255,255,255]
        return cmap
        

    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        instance=False 
        self.pairs = []
        print("Loading data from filesystem...")
        f = open("{}/{}".format(self.data_dir,cfg))               # 返回一个文件对象
        line = f.readline()               # 调用文件的 readline()方法
        f.close()

        images=load_from_multi_page_tiff('{}/train-volume.tif'.format(self.data_dir))
        labels=load_from_multi_page_tiff('{}/train-labels.tif'.format(self.data_dir))

        line = line.split(',')
        indexs = range(to_int(line[0]),to_int(line[1]))

        print("There are {} pictures.".format(len(indexs)))
        print("files name arrays are {}".format(indexs))
        for ids in indexs:
            image = images[ids]
            label = labels[ids]
            if not instance:
                label = (label > 0).astype(np.int)
            mask = np.unique(label)
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))

class DSB(SegDataSet_T):
    def __init__(self,args):
        assert args.target_set is not None,"target_set should not None"
        assert args.data_dir is not None,"data_path should not None"
        SegDataSet_T.__init__(self,args)
        self.target_set = args.target_set
        self.scale = args.scale
        self.vis = False
        helper.DATA_PATH=args.data_dir
    
    def get_classes_names(self):
        return ["background", "Cell"]

    def __call__(self,args):
        train = DSB(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = DSB(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        if False:
            train.statistic_dataset()
            val.statistic_dataset()
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [0,0,0]
        cmap[1] = [255,255,255]
        return cmap
        
    #get pair from png
    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = sample_array("{}/{}".format(self.data_dir,cfg))
        print("There are {} pictures.".format(len(self.data_fns)))
        print("files name arrays are {}".format(self.data_fns))
        for ids in tqdm(self.data_fns):
            #image = np.load('{}/{}/{}.npy'.format(self.data_dir,img,ids ))
            #label = np.load('{}/{}/{}.npy'.format(self.data_dir,GT,ids ))
            image = Image.open('{}/{}/{}{}'.format(self.data_dir,img,ids,self.args.img_ext)).convert("RGB")
            image = np.array(image)
            label =  Image.open('{}/{}/{}{}'.format(self.data_dir,GT,ids,self.args.img_ext)).convert("RGB")
            label = np.array(label)
            label = self.handle_label(label)
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))

class CityScape(SegDataSet_T):
    def __init__(self,args):
        SegDataSet_T.__init__(self,args)
         
    def get_classes_names(self):
        return [
            "road","sidewalk","building","wall","fence","pole","traffic_light","traffic_sign",
            "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
        ]
    
    def __call__(self,args):
        train = CityScape(args)
        train.get_pairs("train.txt")
        train.get_disturibution()
        val = CityScape(args)
        val.get_pairs("test.txt")
        val.get_disturibution()
        n_train = len(train)
        n_val = len(val)
        if False:
            train.statistic_dataset()
            val.statistic_dataset()
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [128, 64, 128]
        cmap[1] = [244, 35, 232]
        cmap[2] = [70, 70, 70]
        cmap[3] = [102, 102, 156]
        cmap[4] = [190, 153, 153]
        cmap[5] = [153, 153, 153]
        cmap[6] = [250, 170, 30]
        cmap[7] = [220, 220, 0]
        cmap[8] = [107, 142, 35]
        cmap[9] = [152, 251, 152]
        cmap[10] = [0, 130, 180]
        cmap[11] = [220, 20, 60]
        cmap[12] = [255, 0, 0]
        cmap[13] = [0, 0, 142]
        cmap[14] = [0, 0, 70]
        cmap[15] = [0, 60, 100]
        cmap[16] = [0, 80, 100]
        cmap[17] = [0, 0, 230]
        cmap[18] = [119, 11, 32]
        return cmap
    
    #get pair from npy
    def get_pairs(self,cfg,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        instance=False 
        self.pairs = []
        print("Loading data from filesystem...")
        f = open("{}/{}".format(self.data_dir,cfg))               # 返回一个文件对象
        line = f.readline()               # 调用文件的 readline()方法
        f.close()
        line = line.split(',')
        indexs = range(to_int(line[0]),to_int(line[1]))
        
        print("There are {} pictures.".format(len(indexs)))
        print("files name arrays are {}".format(indexs))
        for ids in indexs:
            image = np.load('{}/{}/{}.npy'.format(self.data_dir,img,ids ))
            label = np.load('{}/{}/{}.npy'.format(self.data_dir,GT,ids ))
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label,ids))
        
    # map rbg to label
    def rgb2label(self,rgb):
        for index in range(len(self.colors)):
            u =[False,False,False]
            #缩放容错
            u[0] = self.colors[index][0]-8 < rgb[0] and  rgb[0] < self.colors[index][0]+8
            u[1] = self.colors[index][1]-8< rgb[1] and  rgb[1] < self.colors[index][1]+8
            u[2] = self.colors[index][2]-8 < rgb[2] and  rgb[2] < self.colors[index][2]+8
            if u[0] and u[1] and u[2]:
                return self.valid_classes[index]
        return 0


    def __init__(self,args,default_pairs = True):
        self.args = args
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # self.valid_classes = [
        #     7,road
        #     8,
        #     11,
        #     12,
        #     13,
        #     17,
        #     19,
        #     20,
        #     21,
        #     22,
        #     23,
        #     24,
        #     25,
        #     26,
        #     27,
        #     28,
        #     31,
        #     32,
        #     33,
        # ]
        self.valid_classes = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

    def __len__(self):
        return len(self.pairs)

    def init(self,data_dir):
        self.data_dir = data_dir
        self.get_pairs()

    def __call__(self,args):
        t_dataset = CityScapesDataset(args)
        t_dataset.init('{}/train_split'.format(args.data_dir))
        v_dataset = CityScapesDataset(args)
        v_dataset.init('{}/val_split'.format(args.data_dir))
        n_train = len(t_dataset)
        n_val = len(v_dataset)
        train_loader = DataLoader(t_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(v_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader,val_loader,n_train,n_val

    # split label and img from because there only one pic to keep img and label
    def split_image(self,image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    # change label from rbg to class
    def handle_label(self,label):
        label_axes = (label.shape[0],label.shape[1])
        new_label = np.zeros(label_axes).astype(np.uint8)
        for i in range(label.shape[0]):
            for j in range(label.shape[0]):
                new_label[i][j] = self.rgb2label(label[i][j])
        return new_label

    # map rbg to label
    def rgb2label(self,rgb):
        for index in range(len(self.colors)):
            u =[False,False,False]
            #缩放容错
            u[0] = self.colors[index][0]-32 < rgb[0] and  rgb[0] < self.colors[index][0]+32
            u[1] = self.colors[index][1]-32< rgb[1] and  rgb[1] < self.colors[index][1]+32
            u[2] = self.colors[index][2]-32 < rgb[2] and  rgb[2] < self.colors[index][2]+32
            if u[0] and u[1] and u[2]:
                return self.valid_classes[index]
        return 0

    # change label from class label to rbg
    def revert_label2rgb(self, label):
        label_axes = (label.shape[0],label.shape[1],3)
        new_label = np.zeros(label_axes).astype(np.uint8)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                color = self.label2rgb(label[i,j])
                new_label[i,j,:]=color
        return new_label

        # map rbg to label
    def label2rgb(self,label):
        for index in range(len(self.colors)):
            u = (label == self.valid_classes[index])
            if u :
                return self.colors[index]
        return [0,0,0]

    #get pair from npy
    def get_pairs(self,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = os.listdir('{}/{}'.format(self.data_dir,img))
        print("There are {} pictures.".format(len(self.data_fns)))
        for ids in self.data_fns:
            ids = ids[:-4]
            cityscape = np.load('{}/{}/{}.npy'.format(self.data_dir,img,ids ))
            label = np.load('{}/{}/{}.npy'.format(self.data_dir,GT,ids ))
            if imshow:
                self.showrevert(cityscape,label)
            # handle img format
            self.pairs.append((cityscape,label))

    def showrevert(self,cityscape,label):
        rev_label = self.revert_label2rgb(label)
        if cityscape.max()<1.1:
            cityscape *= 255
            cityscape+=128
        if cityscape.shape[0] <4:
            cityscape=cityscape.transpose(1, 2, 0).astype('float32')
        cityscape = Image.fromarray(cityscape.astype(np.uint8))
        rev_label = Image.fromarray(np.uint8(rev_label))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cityscape)
        axes[1].imshow(rev_label)
        plt.show()
    
    def showrevert_cp(self,cityscape,label,pred):
        rev_label = self.revert_label2rgb(label)
        rev_pred = self.revert_label2rgb(pred)
        if cityscape.max()<1.1:
            cityscape *= 255
            cityscape+=128
        if cityscape.shape[0] <4:
            cityscape=cityscape.transpose(1, 2, 0).astype('float32')
        cityscape = Image.fromarray(cityscape.astype(np.uint8))
        rev_label = Image.fromarray(np.uint8(rev_label))
        rev_pred= Image.fromarray(np.uint8(rev_pred))
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(cityscape)
        axes[1].imshow(rev_label)
        axes[2].imshow(rev_pred)
        plt.show()

    def showrevert_cp2file(self,cityscape,label,pred,experiment,epoch=0):
        rev_label = self.revert_label2rgb(label)
        rev_pred = self.revert_label2rgb(pred)
        if cityscape.max()<1.1:
            cityscape *= 255
            cityscape+=128
        if cityscape.shape[0] <4:
            cityscape=cityscape.transpose(1, 2, 0).astype('float32')
        cityscape = Image.fromarray(cityscape.astype(np.uint8))
        rev_label = Image.fromarray(np.uint8(rev_label))
        rev_pred= Image.fromarray(np.uint8(rev_pred))
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(cityscape)
        axes[1].imshow(rev_label)
        axes[2].imshow(rev_pred)
        plt.savefig("{}/{}/{}.png".format('result',experiment,epoch))
        #plt.show()

    # get pair from origin picture
    def get_origin_pairs(self):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = os.listdir(self.data_dir)
        for img_id in tqdm(self.data_fns):
            sample_image_fp = os.path.join(self.data_dir, img_id)
            sample_image = Image.open(sample_image_fp).convert("RGB")
            cityscape,label = self.split_image(sample_image)
            label =  self.handle_label(label)
            self.pairs.append((cityscape,label))

    def __getitem__(self, idx):
        img,label = self.pairs[idx]
        img = img.transpose(2, 0, 1).astype('float32')
        img = (img.astype('float32')-128) / 255
        # onehot = mask2onehot(label,33)
        # to onehot
        return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(label).type(torch.IntTensor),
                'weight':torch.from_numpy(label).type(torch.IntTensor)
        }

    # in preprocess, save img to files
    def pairs2files(self,out_path,img="IMG",GT="GT"):
        i =0
        for cityscape,label in tqdm(self.pairs):
            np.save('{}/{}/{}.npy'.format(out_path,img,i ), cityscape)
            np.save('{}/{}/{}.npy'.format(out_path,GT,i ), label)
            i +=1