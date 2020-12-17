import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import os 


experiment='focal_apd'
os.mkdir("./{}".format(experiment))
filename="./result/{}.csv".format(experiment)
experiment_name ='FOCAL'
pix="focal_apd"
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
    avg_slide=[]
    for id in idx:
        weight_norms.append(csv_data['weight_norm_{}'.format(id)][item])
        gradient_norms.append(csv_data['gradient_norm_{}'.format(id)][item])
        iou.append(csv_data['iou_{}'.format(id)][item])
        logit.append(csv_data['logit_norm_{}'.format(id)][item])
        

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


    plt.title('{}th sample final round detail of {}'.format(item,experiment_name))
    plt.savefig("./{}/{}th_image.png".format(experiment,item))
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


plt.title('mean final round detail of {}'.format(experiment_name))
plt.savefig("./{}/mean_image.png".format(experiment))
    # plt.show()11
plt.cla()
plt.clf()
plt.close()

#--------------------------------------------------------------------------------------------------------------------------------
def print_predict(csv_data,name='avg_slide',showname="mean"):
    for item in range(items):
        slide=[]
        for id in idx:
            slide.append(csv_data['{}_{}'.format(name,id)][item])
        plt.xticks(rotation = 270,fontsize=10)
        plt.plot(categories,slide,'r*-',label='mean')
        plt.legend()
        plt.title('{}th sample {} prediction of {}'.format(item,showname,experiment_name))
        plt.savefig("./{}/{}th_predict_{}.png".format(experiment,item,name))
        # plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    slide=[]
    for id in idx:
        slide.append(np.mean(np.array(csv_data['{}_{}'.format(name,id)])))
    plt.xticks(rotation = 270,fontsize=10)
    plt.plot(categories,slide,'r*-',label='mean')
    plt.legend()
    plt.title('{}th sample {} prediction of {}'.format(item,showname,experiment_name))
    plt.savefig("./{}/mean_predict_{}.png".format(experiment,name))
    
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close()

print_predict(csv_data,name='avg_slide',showname="mean")
print_predict(csv_data,name='val_slide',showname="val")

print_predict(csv_data,name='avg_gt_slide',showname="mean gt")
print_predict(csv_data,name='val_gt_slide',showname="val gt")

print_predict(csv_data,name='avg_ngt_slide',showname="mean ngt")
print_predict(csv_data,name='val_ngt_slide',showname="val ngt")
#--------------------------------------------------------------------------------------------------------------------------------

avg_slide=[]
avg_gt_slide=[]
avg_ngt_slide=[]
iou=[]
for id in idx:
    avg_slide.append(np.mean(np.array(csv_data['avg_slide_{}'.format(id)])))
    avg_gt_slide.append(np.mean(np.array(csv_data['avg_gt_slide_{}'.format(id)])))
    iou.append(np.mean(np.array(csv_data['iou_{}'.format(id)])))
    avg_ngt_slide.append(np.mean(np.array(csv_data['avg_ngt_slide_{}'.format(id)])))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('total and ngt')
ax2 = ax1.twinx()  # this is the important function

l4 = ax2.plot(categories,avg_ngt_slide,'+--',color='#ff9955',label='ngt area')
l1 = ax1.plot(categories,avg_slide,'r*-',label='total')
l2 = ax1.plot(categories,avg_gt_slide,'-.',label='gt area')
l3 = ax1.plot(categories,iou,'co-',label='iou')


fig.legend()
ax2.set_ylabel('ngt')
plt.xlabel('head to tail comparison')
plt.xticks(rotation = 270,fontsize=10)
for tick in ax1.get_xticklabels():
    tick.set_rotation(270)
for tick in ax2.get_xticklabels():
    tick.set_rotation(270)

plt.title('mean final round detail of {}'.format(experiment_name))
plt.savefig("./{}/prediction_mean_total.png".format(experiment))
    # plt.show()11
plt.cla()
plt.clf()
plt.close()


val_slide=[]
val_gt_slide=[]
val_ngt_slide=[]
iou=[]
for id in idx:
    val_slide.append(np.mean(np.array(csv_data['val_slide_{}'.format(id)])))
    val_gt_slide.append(np.mean(np.array(csv_data['val_gt_slide_{}'.format(id)])))
    iou.append(np.mean(np.array(csv_data['iou_{}'.format(id)])))
    val_ngt_slide.append(np.mean(np.array(csv_data['val_ngt_slide_{}'.format(id)])))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('total and ngt')
ax2 = ax1.twinx()  # this is the important function

l4 = ax2.plot(categories,val_ngt_slide,'+--',color='#ff9955',label='ngt area')
l1 = ax1.plot(categories,val_slide,'r*-',label='total')
l2 = ax1.plot(categories,val_gt_slide,'-.',label='gt area')
l3 = ax1.plot(categories,iou,'co-',label='iou')


fig.legend()
ax2.set_ylabel('ngt')
plt.xlabel('head to tail comparison')
plt.xticks(rotation = 270,fontsize=10)
for tick in ax1.get_xticklabels():
    tick.set_rotation(270)
for tick in ax2.get_xticklabels():
    tick.set_rotation(270)

plt.title('var final round detail of {}'.format(experiment_name))
plt.savefig("./{}/prediction_val_total.png".format(experiment))
    # plt.show()11
plt.cla()
plt.clf()
plt.close()