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

__all__ = ['BasicDataset', 'CarvanaDataset','MICDataset','DSBDataset','CityScapesLoader']

class DSBDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
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
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
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
        return img,mask,weight,map_id

class MICDataset(Dataset):
    def __init__(self,target_set=None,data_path=None,scale=1.0,vis = False):
        assert target_set is not None,"target_set should not None"
        assert data_path is not None,"data_path should not None"
        self.target_set = target_set
        self.pairs = []
        self.scale = scale
        self.vis = vis
        helper.DATA_PATH=data_path
        self.get_pairs()


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
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
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
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')


class CityScapesLoader(Dataset):
    def __init__(self,data_dir,default_pairs = True):
        self.data_dir = data_dir
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

        if default_pairs:
            self.get_pairs()

    def __len__(self):
        return len(self.pairs)

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
            if u[0] and u[1] and [2]:
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
        cityscape,label = self.pairs[idx]
        img = cityscape.astype('float32') / 255
        img = cityscape.transpose(2, 0, 1).astype('float32')
        # onehot = mask2onehot(label,33)
        # to onehot
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