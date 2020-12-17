import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def gethex(i):
    if i==0:
        return '0'
    if i==1:
        return '1'
    if i==2:
        return '2'
    if i==3:
        return '3'
    if i==4:
        return '4'
    if i==5:
        return '5'
    if i==6:
        return '6'
    if i==7:
        return '7'
    if i==8:
        return '8'
    if i==9:
        return '9'
    if i==10:
        return 'a'
    if i==11:
        return 'b'
    if i==12:
        return 'c'
    if i==13:
        return 'd'
    if i==14:
        return 'e'
    if i==15:
        return 'f'

def getcolors(count:int):
    k=255-count
    u=int(count/2)
    count1 =int( count /16)
    count2 = count%16
    kc1=int(k/16)
    kc2=k%16
    u1=int(u/16)
    u2=u%16
    return '#{}{}{}{}{}{}'.format(gethex(kc1),gethex(kc2),gethex(u1),gethex(u2),gethex(count1),gethex(count2))


df=pd.read_csv('./results_summary/Cam2007.csv')
#it = df['val_iou_{}'.format('Animal')][0]
#print(it)

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
    summary_factor.append(df['{}'.format(item)][0])

fig, ax = plt.subplots()
summary_factor = np.array(summary_factor )
idx = np.argsort(summary_factor)
summary_factor = np.sort(summary_factor)
names = [class_names[k] for  k in idx]
colors = [cmap[k] for k in idx]
idxs = 31
for i in range(len(class_names)):#total
    ax.bar(names[i], summary_factor[i],color=getcolors(idxs*7))
    idxs-=1
idxs = 31
for a,b in zip(names,summary_factor):
    plt.text(a, b+0, '%.2f' % b, ha='center', va= 'bottom',fontsize=10,color=getcolors(idxs*7))
    idxs -=1
# labels = ax.get_xticklabels()
# for i in range(len(class_names)):#total
#     plt.setp(labels[i], color=RGB_to_Hex("{},{},{}".format(colors[i][0],colors[i][1],colors[i][2])))
plt.xticks(rotation = 300,fontsize=10)
plt.title('Distribution of pixels count')

plt.show()