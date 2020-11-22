import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.tools import file_name

from tqdm import tqdm


# *.csv mIOU in dir
class_names = [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]
target_list=["Road","Building","Tree","Sky","Sidewalk","Fence","Bicyclist"
,"LaneMkgsDriv","Car","Wall","Void","OtherMoving","Pedestrian","TrafficLight","Column_Pole",""
"Truck_Bus","Misc_Text","VegetationMisc","Child"]
def labelcolormap(N):
        cmap = np.zeros((N, 3), dtype = np.uint8)
        cmap[0] = [64,128,64]
        cmap[1] = [192,0,128]
        cmap[2] = [0,128,192]
        cmap[3] = [0,128,64]
        cmap[4] = [128,0,0]
        cmap[5] = [64,0,128]
        cmap[6] = [64,0,192]
        cmap[7] = [192,128,64]
        cmap[8] = [192,192,128]
        cmap[9] = [64,64,128]
        cmap[10] = [128,0,192]
        cmap[11] = [192,0,64]
        cmap[12] = [128,128,64]
        cmap[13] = [192,0,192]
        cmap[14] = [128,64,64]
        cmap[15] = [64,192,128]
        cmap[16] = [64,64,0]
        cmap[17] = [128,64,128]
        cmap[18] = [128,128,192]
        cmap[19] = [0,0,192]
        cmap[20] = [192,128,128]
        cmap[21] = [128,128,128]
        cmap[22] = [64,128,192]
        cmap[23] = [0,0,64]
        cmap[24] = [0,64,64]
        cmap[25] = [192,64,128]
        cmap[26] = [128,128,0]
        cmap[27] = [192,128,192]
        cmap[28] = [64,0,64]
        cmap[29] = [192,192,0]
        cmap[30] = [0,0,0]
        cmap[31] = [64,192,0]
        return cmap
cmap= labelcolormap(32)

dir_name = ("./result/tmp")
file_array = file_name(dir_name)



new_table = dict()
new_table['cases'] = file_array
for item in target_list:
    new_table[item] = []

for file_name in file_array:
    csv_data = pd.read_csv("./result/tmp/{}".format(file_name))
    for item in target_list:
        f1 = round(csv_data['val_iou_{}'.format(item)][29],4)
        new_table[item].append(f1)


df = pd.DataFrame(new_table)
df.to_csv("./results_summary/tail_focal.csv",index=None)



