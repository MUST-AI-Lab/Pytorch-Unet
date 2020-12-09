import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils.dataset as datasets
import argparse
from utils.tools import AverageMeter,str2bool,softmax_helper


experiment='norm_ce_bnout'
filename="./result/{}.csv".format(experiment)
experiment ='NORM CE BNout'
pix="norm_ce_bnout"
epochs = 30
items = 10

weight_pix="{}_weight_norm".format(pix)
gradient_pix="{}_gradient_norm".format(pix)

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

categories = [class_names[x] for x in idx]

for item in range(items):
    weight_norms=[]
    gradient_norms=[]
    iou=[]
    logit=[]
    for id in idx:
        weight_norms.append(csv_data['weight_norm_{}'.format(id)][item])
        iou.append(csv_data['iou_{}'.format(id)][item])
        if val_distribute[item] <1e-9:
            gradient_norms.append(0)
            logit.append(0)
        else:
            gradient_norms.append(csv_data['gradient_norm_{}'.format(id)][item]/(val_distribute[id]*args.input_w*args.input_h))
            logit.append(csv_data['logit_norm_{}'.format(id)][item]/(val_distribute[id]*args.input_w*args.input_h))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('norms and iou')
    ax2 = ax1.twinx()  # this is the important function

    l1 = ax1.plot(categories,weight_norms,'r*-',label='weight norm')
    l2 = ax1.plot(categories,gradient_norms,'-.',label='gradient norm')
    l3 = ax1.plot(categories,iou,'co-',label='iou')
    l4 = ax2.plot(categories,logit,'b+-',label='logit norm')

    fig.legend()
    ax2.set_ylabel('logits norm')
    plt.xlabel('head to tail comparison')
    plt.xticks(rotation = 270,fontsize=10)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(270)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(270)


    plt.title('{}th sample final round detail of {}'.format(item,experiment))
    plt.savefig("{}th_image.png".format(item))
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close()

weight_norms=[]
gradient_norms=[]
iou=[]
logit=[]
for id in idx:
    weight_norms.append(np.mean(np.array(csv_data['weight_norm_{}'.format(id)])))
    iou.append(np.mean(np.array(csv_data['iou_{}'.format(id)])))
    if val_distribute[item] <1e-7:
        gradient_norms.append(0)
        logit.append(0)
    else:
        gradient_norms.append(csv_data['gradient_norm_{}'.format(id)][item]/(val_distribute[id]*args.input_w*args.input_h))
        logit.append(csv_data['logit_norm_{}'.format(id)][item]/(val_distribute[id]*args.input_w*args.input_h))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('norms and iou')
ax2 = ax1.twinx()  # this is the important function

l1 = ax1.plot(categories,weight_norms,'r*-',label='weight norm')
l2 = ax1.plot(categories,gradient_norms,'-.',label='gradient norm')
l3 = ax1.plot(categories,iou,'co-',label='iou')
l4 = ax2.plot(categories,logit,'b+-',label='logit norm')

fig.legend()
ax2.set_ylabel('logits norm')
plt.xlabel('head to tail comparison')
plt.xticks(rotation = 270,fontsize=10)
for tick in ax1.get_xticklabels():
    tick.set_rotation(270)
for tick in ax2.get_xticklabels():
    tick.set_rotation(270)


plt.title('mean final round detail of {}'.format(experiment))
plt.savefig("mean_image.png")
    # plt.show()11
plt.cla()
plt.clf()
plt.close()
