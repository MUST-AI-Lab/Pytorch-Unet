import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.tools import file_name

dir_name = ("./result/tmp")
file_array = file_name(dir_name)

for file_name in file_array:
    csv_data = pd.read_csv("./result/tmp/{}".format(file_name))
    epoch=range(1,31)
    cp1 = csv_data['val_mIOU']
    l1=plt.plot(epoch,cp1,label='{}'.format(file_name[:-4]))

plt.title('loss compartion')
plt.xlabel('epoch')
plt.ylabel('miou')
plt.legend()
plt.show()
