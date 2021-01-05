import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.tools import file_name

dir_name = ("./result/tmp")
#dir_name = ("./result/trace")
x_axis=range(1,31)
title ='ce losses iw'
target_name = 'val_iou_Leak'

file_array = file_name(dir_name)
for file_name in file_array:
    csv_data = pd.read_csv("{}/{}".format(dir_name,file_name))
    cp1 = csv_data[target_name]
    #l1=plt.plot(x_axis,cp1,label='{}'.format(file_name[:-4]),color="#ff9955")
    l1=plt.plot(x_axis,cp1,label='{}'.format(file_name[:-4]))
plt.title('comparison of {}'.format(title))
plt.xlabel('epoch')
plt.ylabel('miou')
plt.legend()
plt.show()
