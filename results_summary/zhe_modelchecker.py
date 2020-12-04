import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd


experiment='ce_tdev2'
filename="./result/{}.csv".format(experiment)
experiment ='TDE v2'
pix="TDEv2"
epochs = 30
items = 10

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

categories = [class_names[x] for x in idx]

for item in range(items):
    weight_norms=[]
    gradient_norms=[]
    iou=[]
    logit=[]
    for id in idx:
        weight_norms.append(csv_data['weight_norm_{}'.format(id)][item])
        gradient_norms.append(csv_data['gradient_norm_{}'.format(id)][item])
        iou.append(csv_data['iou_{}'.format(id)][item])
        logit.append(csv_data['logit_norm_{}'.format(id)][item])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('norms and iou')
    ax2 = ax1.twinx()  # this is the important function

    l1 = ax1.plot(categories,weight_norms,'b*-',label='weight norm')
    l2 = ax1.plot(categories,gradient_norms,'-.',label='gradient norm')
    l3 = ax1.plot(categories,iou,'co-',label='iou')
    l4 = ax2.plot(categories,logit,'r+-',label='logit norm')

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
    gradient_norms.append(np.mean(np.array(csv_data['gradient_norm_{}'.format(id)])))
    iou.append(np.mean(np.array(csv_data['iou_{}'.format(id)])))
    logit.append(np.mean(np.array(csv_data['logit_norm_{}'.format(id)])))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('norms and iou')
ax2 = ax1.twinx()  # this is the important function

l1 = ax1.plot(categories,weight_norms,'b*-',label='weight norm')
l2 = ax1.plot(categories,gradient_norms,'-.',label='gradient norm')
l3 = ax1.plot(categories,iou,'co-',label='iou')
l4 = ax2.plot(categories,logit,'r+-',label='logit norm')

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
