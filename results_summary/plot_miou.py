import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.tools import file_name

dir_name = ("./result/tmp")
#dir_name = ("./result/trace")
x_axis=range(1,31)
target_name = 'val_mIOU'

file_array = file_name(dir_name)
for file_name in file_array:
    csv_data = pd.read_csv("{}/{}".format(dir_name,file_name))
    cp1 = csv_data[target_name][29]
    print("{}:{}={}".format(file_name,target_name,cp1))

