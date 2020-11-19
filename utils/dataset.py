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
from utils.tools import mask2onehot,onehot2mask
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from utils.weights_collate import default_collate_with_weight,label2_baseline_weight_by_prior,label2distribute,distribution2tensor
import pandas as pd

__all__ = ['BasicDataset', 'CarvanaDataset','MICDataset','DSBDataset','CityScapesDataset','PascalDataset','Cam2007Dataset']
def RGB_to_Hex(tmp):
    rgb = tmp.split(',')#将RGB格式划分开来
    strs = '#'
    for i in rgb:
        num = int(i)#将str转int
        #将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x','0').upper()
    return strs


class Cam2007Dataset(Dataset):
    def __init__(self,args,default_pairs = True):
        self.data_dir =args.data_dir
        self.args = args
        self.cmap = self.labelcolormap(32)
        self.class_names = [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]

    # get pixel distribution from data set
    def statistic_dataset(self):
        keeper = dict()
        keeper['id'] = []
        for item in self.class_names:
            keeper[item] = []
        for _,label,ids in tqdm(self.pairs):
            keeper['id'].append(ids)
            total = np.prod(label.shape)
            total_pixel =0
            for i in range(32):
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
            ax.bar(names[i], summary_factor[i],color=RGB_to_Hex("{},{},{}".format(colors[i][0],colors[i][1],colors[i][2])))
        for a,b in zip(names,summary_factor):
            plt.text(a, b+0, '%.2f' % b, ha='center', va= 'bottom',fontsize=10)
        plt.xticks(rotation = 270,fontsize=10)
        plt.title('Distribution of pixels count')
        plt.show()

        dataframe = pd.DataFrame.from_dict(keeper)
        dataframe.to_csv("Cam2007Dataset.csv",index=False)

    def __call__(self,args):
        dataset = Cam2007Dataset(args)
        dataset.get_pairs()
        dataset.get_disturibution()
        n_val = int(len(dataset) * args.val)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=default_collate_with_weight)
        return train_loader,val_loader,n_train,n_val

    def __len__(self):
        return len(self.pairs)

    def uint82bin(self, n, count=8):
        return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

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

    # change label from rbg to class
    def handle_label(self,label):
        label_axes = (label.shape[0],label.shape[1])
        new_label = np.zeros(label_axes).astype(np.uint8)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                new_label[i][j] = self.rgb2label(label[i][j])
        return new_label

    # map rbg to label
    def rgb2label(self,rgb):
        for index in range(len(self.cmap)):
            u = rgb==self.cmap[index]
            if u[0] and u[1] and u[2]:
                return index
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

        # map rbg to label
    def label2rgb(self,label):
        return self.cmap[label]

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
        self.summary_factor = summary_factor

    #get pair from npy
    def get_pairs(self,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = os.listdir('{}/{}'.format(self.data_dir,img))
        print("There are {} pictures.".format(len(self.data_fns)))
        for ids in self.data_fns:
            ids = ids[:-4]
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

    def showrevert(self,cityscape,label):
        rev_label = self.revert_label2rgb(label)
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

    def showrevert_cp2(self,cityscape,label,pred):
        rev_label = label
        rev_pred = self.revert_label2rgb(pred)
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

    def __getitem__(self, idx):
        cityscape,label,_ = self.pairs[idx]
        img = cityscape.astype('float32') / 255
        img = cityscape.transpose(2, 0, 1).astype('float32')
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

    # get pair from origin picture
    def get_origin_pairs(self,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = os.listdir('{}/{}'.format(self.data_dir,img))
        print("There are {} pictures.".format(len(self.data_fns)))
        for ids in tqdm(self.data_fns):
            ids = ids[:-4]
            image =  Image.open('{}/{}/{}.png'.format(self.data_dir,img,ids )).convert("RGB")
            label = Image.open('{}/{}/{}_L.png'.format(self.data_dir,GT,ids )).convert("RGB")
            image = np.array(image)
            new_label = np.array(label)
            new_label = self.handle_label(new_label)
            if imshow:
                self.showrevert_cp2(image,label,new_label)
            # handle img format
            self.pairs.append((image,new_label))

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

class DSBDataset(torch.utils.data.Dataset):
    def __init__(self,args, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.args =args

    def __call__(self,args):
        # Data loading code
        img_ids = glob(os.path.join('data', args.data_dir, 'images', '*' + args.img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=args.val, random_state=41)

        train_transform = Compose([
            transforms.RandomRotate90(),
            transforms.Flip(),
            OneOf([
                transforms.HueSaturationValue(),
                transforms.RandomBrightness(),
                transforms.RandomContrast(),
            ], p=1),
            transforms.Resize(args.input_h, args.input_w),
            transforms.Normalize(),
        ])

        val_transform = Compose([
            transforms.Resize(args.input_h, args.input_w),
            transforms.Normalize(),
        ])
        train_dataset = DSBDataset(args)
        train_dataset.init(
            img_ids=train_img_ids,
            img_dir=os.path.join('data', args.data_dir, 'images'),
            mask_dir=os.path.join('data', args.data_dir, 'masks'),
            img_ext=args.img_ext,
            mask_ext=args.mask_ext,
            num_classes=args.num_classes,
            transform=None)
        val_dataset =  DSBDataset(args)
        val_dataset.init(
            img_ids=val_img_ids,
            img_dir=os.path.join('data', args.data_dir, 'images'),
            mask_dir=os.path.join('data', args.data_dir, 'masks'),
            img_ext=args.img_ext,
            mask_ext=args.mask_ext,
            num_classes=args.num_classes,
            transform=None)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        return train_loader,val_loader,len(train_dataset),len(val_dataset)

    def init(self,img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir =mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.pairs = []
        self.get_pairs()

    def __len__(self):
        return len(self.pairs)

    def get_pairs(self):
        print("Loading data from filesystem...")
        for img_id in self.img_ids:
            img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
            mask = []
            weight = []
            for i in range(self.num_classes):
                tmp = (cv2.imread(os.path.join(self.mask_dir, str(i),
                            img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]>0).astype(np.float32)
                # for scale between 0-255
                tmp = np.squeeze(tmp,axis=-1)
                mask.append(tmp)
                weight.append(helper.weight_map(tmp))
            mask = np.dstack(mask)
            weight = np.dstack(weight)
            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            img = img.astype('float32') / 255

            img = img.transpose(2, 0, 1).astype('float32')
            mask = mask.transpose(2, 0, 1).astype('float32')
            weight = weight.transpose(2, 0, 1).astype('float32')
            self.pairs.append((img, mask,weight, {'img_id': img_id}))


    def __getitem__(self, idx):
        img,mask,weight,map_id = self.pairs[idx]
        return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'weight':torch.from_numpy(weight).type(torch.FloatTensor)
            }

class MICDataset(Dataset):
    def __init__(self,args,vis = False):
        assert args.target_set is not None,"target_set should not None"
        assert args.data_dir is not None,"data_path should not None"
        self.target_set = args.target_set
        self.pairs = []
        self.scale = args.scale
        self.vis = vis
        helper.DATA_PATH=args.data_dir

    def init(self):
        self.get_pairs()

    def __call__(self,args):
        dataset =MICDataset(args)
        dataset.init()
        n_val = int(len(dataset) * args.val)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        return train_loader,val_loader,n_train,n_val

    def __len__(self):
        return len(self.pairs)

    @classmethod
    def preprocess(cls, img, scale):
        pil_img = Image.fromarray(img)
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def get_pairs(self):
        print("Loading data from filesystem...")
        X_0, Y_0 = helper.get_data(self.target_set)
        W_0 = helper.get_weight_maps(Y_0)
        datas = zip(X_0,Y_0,W_0)
        for item in datas:
            self.pairs.append(item)

    def __getitem__(self, i):
        img,mask,weight = self.pairs[i]
        if self.scale >0:
            img = self.preprocess(img,self.scale)
            mask = self.preprocess(mask,self.scale)
            weight = self.preprocess(weight,self.scale)

            if  self.vis:
                helper.visualize_pred(img, mask, weight, title1="Original", title2="True", title3="WeightMap")
            # for channel first: 1 channel
            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'weight':torch.from_numpy(weight).type(torch.FloatTensor)
            }
        else:# for channel fitting
            if  self.vis:
                helper.visualize_pred(img, mask, weight, title1="Original", title2="True", title3="WeightMap")
            # for channel first: 1 channel
            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor).unsqueeze(0),
                'weight':torch.from_numpy(weight).type(torch.FloatTensor).unsqueeze(0)
            }

class BasicDataset(Dataset):
    def __init__(self, args):
        self.imgs_dir = args.imgs_dir
        self.masks_dir = args.masks_dir
        self.scale = args.scale
        self.mask_suffix = args.mask_suffix
        assert 0 < args.scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(args.imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, args):
        super().__init__(args)

class PascalDataset(Dataset):
    def __init__(self,args,default_pairs = True):
        self.data_dir = args.data_dir
        self.cmap = self.labelcolormap(21)
        self.class_names = [
            "B-ground", "Aeroplane","Bicycle","Bird","Boat","Bottle","Bus",
            "Car","Cat","Chair","Cow","Dining-Table","Dog","Horse",
            "Motobike","Person","Potted-Plant","Sheep","Sofa","Train","TV/Monitor"
        ]

    def __len__(self):
        return len(self.pairs)

    def __call__(self,args):
        dataset = PascalDataset(args)
        dataset.get_pairs()
        n_val = int(len(dataset) * args.val)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader,val_loader,n_train,n_val

    def uint82bin(self, n, count=8):
        return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

    def labelcolormap(self,N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        for i in range(N):
            r = 0
            g = 0
            b = 0
            id = i
            for j in range(7):
                str_id = self.uint82bin(id)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        # cmap[:] = cmap[::-1]
        return cmap

    # change label from rbg to class
    def handle_label(self,label):
        label_axes = (label.shape[0],label.shape[1])
        new_label = np.zeros(label_axes).astype(np.uint8)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                new_label[i][j] = self.rgb2label(label[i][j])
        return new_label

    # map rbg to label
    def rgb2label(self,rgb):
        for index in range(len(self.cmap)):
            u = rgb==self.cmap[index]
            if u[0] and u[1] and u[2]:
                return index
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
        return self.cmap[label]

    #get pair from npy
    def get_pairs(self,img="IMG",GT="GT",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = os.listdir('{}/{}'.format(self.data_dir,img))
        print("There are {} pictures.".format(len(self.data_fns)))
        for ids in self.data_fns:
            ids = ids[:-4]
            image = np.load('{}/{}/{}.npy'.format(self.data_dir,img,ids ))
            label = np.load('{}/{}/{}.npy'.format(self.data_dir,GT,ids ))
            if imshow:
                rev_label = self.revert_label2rgb(label)
                image,rev_label = Image.fromarray(image), Image.fromarray(np.uint8(rev_label))
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[1].imshow(rev_label)
                plt.show()
            self.pairs.append((image,label))

    def showrevert(self,cityscape,label):
        rev_label = self.revert_label2rgb(label)
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

    def showrevert_cp2(self,cityscape,label,pred):
        rev_label = label
        rev_pred = self.revert_label2rgb(pred)
        if cityscape.max()<1:
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

    def __getitem__(self, idx):
        img,label = self.pairs[idx]
        img = img.transpose(2, 0, 1).astype('float32')
        img = (img.astype('float32')-128) / 255
        # onehot = mask2onehot(label,33)
        # to onehot
        return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(label).type(torch.IntTensor)
        }

    # get pair from origin picture
    def get_origin_pairs(self,img="JPEGImages",GT="SegmentationClass",imshow=False):
        self.pairs = []
        print("Loading data from filesystem...")
        self.data_fns = os.listdir('{}/{}'.format(self.data_dir,img))
        print("There are {} pictures.".format(len(self.data_fns)))
        for ids in tqdm(self.data_fns):
            ids = ids[:-4]
            image =  Image.open('{}/{}/{}.jpg'.format(self.data_dir,img,ids )).convert("RGB")
            label = Image.open('{}/{}/{}.png'.format(self.data_dir,GT,ids )).convert("RGB")
            image = np.array(image)
            new_label = np.array(label)
            new_label = self.handle_label(new_label)
            if imshow:
                self.showrevert_cp2(image,label,new_label)
            # handle img format
            self.pairs.append((image,new_label))

    # in preprocess, save img to files
    def pairs2files(self,out_path,img="IMG",GT="GT"):
        i =0
        for cityscape,label in tqdm(self.pairs):
            np.save('{}/{}/{}.npy'.format(out_path,img,i ), cityscape)
            np.save('{}/{}/{}.npy'.format(out_path,GT,i ), label)
            i +=1

class CityScapesDataset(Dataset):
    def __init__(self,args,default_pairs = True):
        self.args = args
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # self.valid_classes = [
        #     7,
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