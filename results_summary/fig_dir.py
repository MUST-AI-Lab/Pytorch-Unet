import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.tools import file_name

from tqdm import tqdm

class_names = [
            "Animal", "Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
            "Child","Column_Pole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter",
            "OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol",
            "Sky","SUVPickupTruck","TrafficCone","TrafficLight","Train","Tree","Truck_Bus","Tunnel",
            "VegetationMisc","Void","Wall"
        ]
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

for file_name in tqdm(file_array):
    df=pd.read_csv('{}/{}'.format(dir_name,file_name))

    def RGB_to_Hex(tmp):
        rgb = tmp.split(',')#将RGB格式划分开来
        strs = '#'
        for i in rgb:
            num = int(i)#将str转int
            #将R、G、B分别转化为16进制拼接转换并大写
            strs += str(hex(num))[-2:].replace('x','0').upper()
        return strs

    summary_factor=[]
    for item in class_names:
        summary_factor.append(df['val_iou_{}'.format(item)][29])

    plt.figure()
    fig, ax = plt.subplots( figsize=(13, 5))
    summary_factor = np.array(summary_factor )
    idx = np.argsort(summary_factor)
    summary_factor = np.sort(summary_factor)
    names = [class_names[k] for  k in idx]
    colors = [cmap[k] for k in idx]
    for i in range(len(class_names)):#total
        ax.bar(names[i], summary_factor[i],color=RGB_to_Hex("{},{},{}".format(colors[i][0],colors[i][1],colors[i][2])))
    for a,b in zip(names,summary_factor):
        plt.text(a, b+0, '%.2f' % b, ha='center', va= 'bottom',fontsize=10)
    plt.xticks(rotation = 270,fontsize=10)
    plt.title('The iou distribution  in {}'.format(file_name[:-4]))
    #plt.show()
    plt.savefig('{}/result/{}.png'.format(dir_name,file_name))
    plt.cla()
    plt.clf()