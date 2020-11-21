import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
# momentum=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
# ce=[0.3497,0.3556,0.3692,0.3716,0.3735,0.3752,0.3756,0.4223,0.4302,0.4277,0.3945,0.4052,0.4136,0.3354,0.3617,0.3019,0.3682,0.3226,0.3182]
# focal=[0.3200,0.3152,0.3315,0.3574,0.3633,0.3891,0.3815,0.3868,0.3988,0.4194,0.3921,0.4223,0.4284,0.4019,0.4061,0.3218,0.4140,0.2859,0.3852]

class_names = [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]
target=[17,4,26,21,19,9,2,11,5,31,30,14]
csv_data = pd.read_csv("./result/trainv2_focal_baseline_shuffle.csv")
momentum=range(0,30)

print(csv_data)
for item in target:
    cp1 = csv_data['train_loss_{}'.format(item)]
    l1=plt.plot(momentum,cp1,label='{}'.format(class_names[item]))

plt.title('head classes training loss')
plt.xlabel('epoch')
plt.ylabel('focal loss baseline weight')
plt.legend()
plt.show()

plt.cla()
plt.clf()

tail_target = range(32)

for item in tail_target:
    if item not in target:
        cp1 = csv_data['train_loss_{}'.format(item)]
        l1=plt.plot(momentum,cp1,label='{}'.format(class_names[item]))

plt.title('tail classes training loss')
plt.xlabel('epoch')
plt.ylabel('focal loss baseline weight')
plt.legend()
plt.show()

loss_detail ={}
for i in range(32):
    loss_detail[class_names[i]] = csv_data['train_loss_{}'.format(i)][29]

df = pd.DataFrame(data=loss_detail,index = [0])
df.to_csv("detail_loss_trace_ce.csv",index=False)

