import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils.dataset as datasets
import argparse
from utils.tools import AverageMeter,str2bool,softmax_helper

experiment='norm_ce_bnout'
filename="./result/trace/{}.csv".format(experiment)
init_filename = "./result/trace/{}_init.csv".format(experiment)
lossname ='NORM CE BNout '
pix="norm_ce_bnout"
epochs = 30

weight_norm_kind = 'train_final_norm'
gradient_norm_kind = 'train_loss_gd_norm'
weight_norm_name = "final layer weight norm"
gradient_norm_name = "final layer gradient norm"

DATASET_NAMES = datasets.__all__
def get_args():
    # dataset
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('--dataset', metavar='DATASET', default='Cam2007DatasetV2',
                        choices=DATASET_NAMES,
                        help='model architecture: ' +
                        ' | '.join(DATASET_NAMES) +
                        ' (default: BasicDataset)')
    parser.add_argument('--data_dir', default='./data/Cam2007_n',
                        help='dataset_location_dir')
    parser.add_argument('--num_workers', default=0, type=int)
    #for dsb dataset compact
    parser.add_argument('--input_w', default=960, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=720, type=int,
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

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=32, type=int,
                        help='number of classes')
    return parser.parse_args()

args=get_args()
datamaker = datasets.__dict__[args.dataset](args)
train_loader,val_loader,n_train,n_val = datamaker(args)

val_distribute = val_loader.dataset.summary_factor
val_distribute = np.array(val_distribute)

weight_pix="{}_weight_norm".format(pix)
gradient_pix="{}_gradient_norm".format(pix)

#plot weight------------------------------------------------------------------------------------
idx = [17, 4, 26, 21, 19, 9, 2, 10, 5, 31, 30, 14, 16, 24, 8, 27, 12, 29, 7, 20, 1, 6, 11, 0, 3, 13, 15, 18, 22, 23, 25, 28]
class_names = [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]
csv_data = pd.read_csv(filename)
init_csv_data = pd.read_csv(init_filename)
epoch = csv_data['epoch']

print(csv_data)
final_epoch = []
for item in idx:
    cp1 = csv_data['{}_{}'.format(weight_norm_kind,item)]
    l1=plt.plot(epoch,cp1,label='{}'.format(class_names[item]))
    final_epoch.append(csv_data['{}_{}'.format(weight_norm_kind,item)][len(epoch)-1])

plt.title('{} of {} loss'.format(weight_norm_name,lossname))
plt.xlabel('epoch')
plt.ylabel('')
plt.legend(framealpha=0.5)
# plt.show()
plt.savefig("{}_zhe.png".format(weight_pix))

plt.cla()
plt.clf()

categories = [class_names[x] for x in idx]
l1=plt.plot(categories,final_epoch,'ro-',label='{} of last epoch'.format(weight_norm_name))

plt.title('30th epoch:{} in {} loss'.format(weight_norm_name,lossname))
plt.xlabel('head to tail categories')
plt.xticks(rotation = 270,fontsize=10)
plt.ylabel('')
plt.legend(framealpha=0.5)
# plt.show()
plt.savefig("{}_last.png".format(weight_pix))

plt.cla()
plt.clf()
#----------------------------------------------------------------------plot weight norm for  each epoch
norm_epoch = []
for item in idx:
    norm_epoch.append(init_csv_data['{}_{}'.format(weight_norm_kind,item)][0])
categories = [class_names[x] for x in idx]
l1=plt.plot(categories,norm_epoch,'ro-',label='{} of last epoch'.format(weight_norm_name))
plt.title('init {} in {} loss'.format(weight_norm_name,lossname))
plt.xlabel('head to tail categories')
plt.xticks(rotation = 270,fontsize=10)
plt.ylabel('')
plt.legend(framealpha=0.5)
plt.savefig("{}_init_epoch.png".format(weight_pix))
plt.cla()
plt.clf()
#----------------------------------------------------------------------init in upper
for e in range(epochs):
    norm_epoch = []
    for item in idx:
        norm_epoch.append(csv_data['{}_{}'.format(weight_norm_kind,item)][e])
    categories = [class_names[x] for x in idx]
    l1=plt.plot(categories,norm_epoch,'ro-',label='{} of last epoch'.format(weight_norm_name))
    plt.title('{}th epoch:{} in {} loss'.format((e+1),weight_norm_name,lossname))
    plt.xlabel('head to tail categories')
    plt.xticks(rotation = 270,fontsize=10)
    plt.ylabel('')
    plt.legend(framealpha=0.5)
    plt.savefig("{}_{}th_epoch.png".format(weight_pix,e))
    plt.cla()
    plt.clf()

plt.cla()
plt.clf()

fig = plt.figure()  # 创建画布
ax = plt.subplot()  # 创建作图区域
ax.set_title('Comparison of {} each category  during 30 epochs'.format(weight_norm_name))
# 蓝色矩形的红线：50%分位点是4.5,上边沿：25%分位点是2.25,下边沿：75%分位点是6.75
regions = []
for item in idx:
    regions.append(sorted(csv_data['{}_{}'.format(weight_norm_kind,item)]))
ax.boxplot(regions)
ax.set_xticklabels(categories,
                    rotation=45, fontsize=8)
#ax.plot(categories,final_epoch,'ro--',label='weight norm of last epoch')


plt.xticks(rotation = 270,fontsize=10)
plt.legend(framealpha=0.5)
# plt.show()
plt.savefig("{}_box.png".format(weight_pix))


plt.cla()
plt.clf()

#plot gradient------------------------------------------------------------------------------------
idx = [17, 4, 26, 21, 19, 9, 2, 10, 5, 31, 30, 14, 16, 24, 8, 27, 12, 29, 7, 20, 1, 6, 11, 0, 3, 13, 15, 18, 22, 23, 25, 28]
class_names = [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]
csv_data = pd.read_csv(filename)
epoch = csv_data['epoch']

print(csv_data)
final_epoch = []
for item in idx:
    cp1 = csv_data['{}_{}'.format(gradient_norm_kind,item)]
    if val_distribute[item] <1e-7:
        cp1 =  np.zeros_like(cp1)
        final_epoch.append(0)
    else:
        cp1 = np.array(cp1)/(val_distribute[item]*args.input_w*args.input_h)
        final_epoch.append(csv_data['{}_{}'.format(gradient_norm_kind,item)][len(epoch)-1]/(val_distribute[item]*args.input_w*args.input_h))
    l1=plt.plot(epoch,cp1,label='{}'.format(class_names[item]))

plt.title('{} of {} loss'.format(gradient_norm_name,lossname))
plt.xlabel('epoch')
plt.ylabel('')
plt.legend(framealpha=0.5)
# plt.show()
plt.savefig("{}_zhe.png".format(gradient_pix))

plt.cla()
plt.clf()

categories = [class_names[x] for x in idx]
l1=plt.plot(categories,final_epoch,'ro-',label='{} of last epoch'.format(gradient_norm_name))

plt.title('30th epoch:{} in {} loss'.format(gradient_norm_name,lossname))
plt.xlabel('head to tail categories')
plt.xticks(rotation = 270,fontsize=10)
plt.ylabel('')
plt.legend(framealpha=0.5)
# plt.show()
plt.savefig("{}_last.png".format(gradient_pix))

plt.cla()
plt.clf()

fig = plt.figure()  # 创建画布
ax = plt.subplot()  # 创建作图区域
ax.set_title('Comparison of {} each category  during 30 epochs'.format(gradient_norm_name))
# 蓝色矩形的红线：50%分位点是4.5,上边沿：25%分位点是2.25,下边沿：75%分位点是6.75
regions = []
for item in idx:
    regions.append(sorted(csv_data['{}_{}'.format(gradient_norm_kind,item)]))
ax.boxplot(regions)
ax.set_xticklabels(categories,
                    rotation=45, fontsize=8)
#ax.plot(categories,final_epoch,'ro--',label='weight norm of last epoch')


plt.xticks(rotation = 270,fontsize=10)
plt.legend(framealpha=0.5)
# plt.show()
plt.savefig("{}_box.png".format(gradient_pix))