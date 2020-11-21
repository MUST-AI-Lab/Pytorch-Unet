import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd


filename="./result/trace/trainv2_focal.csv"
lossname ='focal'
pix="focal"

weight_norm_kind = 'train_final_norm'
gradient_norm_kind = 'train_loss_gd_norm'
weight_norm_name = "final layer weight norm"
gradient_norm_name = "final layer gradient norm"


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
    l1=plt.plot(epoch,cp1,label='{}'.format(class_names[item]))
    final_epoch.append(csv_data['{}_{}'.format(gradient_norm_kind,item)][len(epoch)-1])

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