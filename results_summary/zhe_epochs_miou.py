import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.tools import file_name

#dir_name = ("./result/tmp")
dir_name = ("./result/trace")
x_axis=range(1,31)
title ='ce losses with baseline'
target_name = 'val_mIOU'

file_array = file_name(dir_name)
for file_name in file_array:
    csv_data = pd.read_csv("./result/trace/{}".format(file_name))
    cp1 = csv_data[target_name]
    l1=plt.plot(x_axis,cp1,label='{}'.format(file_name[:-4]))

plt.title('comparison of {}'.format(title))
plt.xlabel('epoch')
plt.ylabel('miou')
plt.legend()
plt.show()
