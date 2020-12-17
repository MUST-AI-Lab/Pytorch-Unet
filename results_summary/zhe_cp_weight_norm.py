import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd

lossname ='norm last layer dice'
experiments=['dice','dice_baseline']
lines=['ro-','b+-']
names=['dice','dice_iw']
pix="cp_dice"
epochs = 30
weight_pix="{}_weight_norm".format(pix)

for experiment,line,name in zip(experiments,lines,names):
    filename="./result/trace/{}.csv".format(experiment)
    init_filename = "./result/trace/{}_init.csv".format(experiment)



    weight_norm_kind = 'train_final_norm'
    gradient_norm_kind = 'train_loss_gd_norm'
    weight_norm_name = "final layer weight norm"
    gradient_norm_name = "final layer gradient norm"

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
        final_epoch.append(csv_data['{}_{}'.format(weight_norm_kind,item)][len(epoch)-1])

    categories = [class_names[x] for x in idx]
    l1=plt.plot(categories,final_epoch,line,label=name)

plt.title('30th epoch:{} in {} loss'.format(weight_norm_name,lossname))
plt.xlabel('head to tail categories')
plt.xticks(rotation = 270,fontsize=10)
plt.ylabel('')
plt.legend(framealpha=0.5)
# plt.show()
plt.savefig("{}_last.png".format(weight_pix))

plt.cla()
plt.clf()
