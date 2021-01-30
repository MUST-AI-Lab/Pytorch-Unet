import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.tools import file_name

file_name = "./result/default_ce.csv"
file_name2 = "./result/default.csv"
csv_data = pd.read_csv(file_name)
csv_data2 = pd.read_csv(file_name2)

name1="ce"
name2="ce_seesaw"

print(csv_data.head())
print(csv_data2.head())

total_epoch =30
epochs = range(0,total_epoch)
categorys=2

# idx=[1,0,2,3]
# class_names=["Key","Background","Key_Light","Leak"]

idx=[0,1]
class_names=["Background","Cell"]

# idx = [17, 4, 26, 21, 19, 9, 2, 10, 5, 31, 30, 14, 16, 24, 8, 27, 12, 29, 7, 20, 1, 6, 11, 0, 3, 13, 15, 18, 22, 23, 25, 28]
# class_names = [
#             "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
#             "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
#             "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
#             "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
#             "VegetationMisc","Void","Wall"
#         ]

#for each
for cate in range(categorys):
    cp1 = csv_data['train_positive_gd_cumulative_{}'.format(cate)]
    cp2 = csv_data['train_negative_gd_cumulative_{}'.format(cate)]
    ac_positive = []
    ac_negative = []
    for i in range(total_epoch):
        ac_positive.append(np.sum(cp1[0:i+1]))
        ac_negative.append(np.sum(cp2[0:i+1]))
    radio1 = np.array(ac_positive)/np.array(ac_negative)
    radio1 = np.array(radio1)
    l1=plt.plot(epochs,radio1,'r-',label='radio_{}_category_{}'.format(name1,class_names[cate]))

    cp1 = csv_data2['train_positive_gd_cumulative_{}'.format(cate)]
    cp2 = csv_data2['train_negative_gd_cumulative_{}'.format(cate)]
    ac_positive = []
    ac_negative = []
    for i in range(total_epoch):
        ac_positive.append(np.sum(cp1[0:i+1]))
        ac_negative.append(np.sum(cp2[0:i+1]))
    radio2 = np.array(ac_positive)/np.array(ac_negative)
    radio2 = np.array(radio2)
    l2=plt.plot(epochs,radio2,'b--',label='radio_{}_category_{}'.format(name2,class_names[cate]))

    plt.title('Ratio of cumulative gradients between positive samples\n and negative samples \n for {} category'.format(class_names[cate]))
    plt.xlabel('epochs')
    plt.ylabel('gradient')
    plt.legend()
    plt.savefig("radio_{}.png".format(cate))
    plt.cla()
    plt.clf()
    plt.close()

categories = [class_names[x] for x in idx]

radios = []
for cate in idx:
    cp1 = csv_data['train_positive_gd_cumulative_{}'.format(cate)]
    cp2 = csv_data['train_negative_gd_cumulative_{}'.format(cate)]
    radios.append(np.sum(cp1)/np.sum(cp2))
radios = np.array(radios)
l1=plt.plot(categories,radios,'r-',label='radios_{}'.format(name1))

radios2 = []
for cate in idx:
    cp1 = csv_data2['train_positive_gd_cumulative_{}'.format(cate)]
    cp2 = csv_data2['train_negative_gd_cumulative_{}'.format(cate)]
    radios2.append(np.sum(cp1)/np.sum(cp2))
radios2 = np.array(radios2)
l1=plt.plot(categories,radios2,'b--',label='radios_{}'.format(name2))

plt.title('Ratio of cumulative gradients between positive samples\n and negative samples \n for each category')
plt.xlabel('epochs')
plt.xticks(rotation = 270,fontsize=10)
plt.ylabel('gradient_radio')
plt.legend()
plt.savefig("total_radio.png")
plt.cla()
plt.clf()
plt.close()

