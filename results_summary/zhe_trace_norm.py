import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
# momentum=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
# ce=[0.3497,0.3556,0.3692,0.3716,0.3735,0.3752,0.3756,0.4223,0.4302,0.4277,0.3945,0.4052,0.4136,0.3354,0.3617,0.3019,0.3682,0.3226,0.3182]
# focal=[0.3200,0.3152,0.3315,0.3574,0.3633,0.3891,0.3815,0.3868,0.3988,0.4194,0.3921,0.4223,0.4284,0.4019,0.4061,0.3218,0.4140,0.2859,0.3852]

idx = [17, 4, 26, 21, 19, 9, 2, 10, 5, 31, 30, 14, 16, 24, 8, 27, 12, 29, 7, 20, 1, 6, 11, 0, 3, 13, 15, 18, 22, 23, 25, 28]
class_names = [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]
csv_data = pd.read_csv("./result/default.csv")
epoch = csv_data['epoch']

print(csv_data)
final_epoch = []
for item in idx:
    cp1 = csv_data['train_final_norm_{}'.format(item)]
    l1=plt.plot(epoch,cp1,label='{}'.format(class_names[item]))
    final_epoch.append(csv_data['train_final_norm_{}'.format(item)][9])

plt.title('weight_norm')
plt.xlabel('epoch')
plt.ylabel('weight_norm of ce loss')
plt.legend()
plt.show()

plt.cla()
plt.clf()

categories = range(32)
l1=plt.plot(categories,final_epoch,label='weight norm var class')

plt.title('weight_norm')
plt.xlabel('head to tail categories')
plt.ylabel('weight_norm of ce loss')
plt.legend()
plt.show()
