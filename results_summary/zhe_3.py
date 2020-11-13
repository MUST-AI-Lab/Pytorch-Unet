import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
# momentum=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
# ce=[0.3497,0.3556,0.3692,0.3716,0.3735,0.3752,0.3756,0.4223,0.4302,0.4277,0.3945,0.4052,0.4136,0.3354,0.3617,0.3019,0.3682,0.3226,0.3182]
# focal=[0.3200,0.3152,0.3315,0.3574,0.3633,0.3891,0.3815,0.3868,0.3988,0.4194,0.3921,0.4223,0.4284,0.4019,0.4061,0.3218,0.4140,0.2859,0.3852]

csv_data = pd.read_csv("./results_summary/weight_momentum.csv")
momentum=csv_data['mon']
cp1 = csv_data['wuzhou_focal']
cp2 = csv_data['wuzhou_focal_weight']
cp3 = csv_data['wuzhou_focal_weight_local']
cp4 = csv_data['wuzhou_focal_baseline_global']
cp5 = csv_data['wuzhou_focal_baseline']
# cp6 = csv_data['focal']
# cp7 = csv_data['dicev2_weight']
# cp8 = csv_data['dicev2']

l1=plt.plot(momentum,cp1,'ro-',label='focal without weight')
l2=plt.plot(momentum,cp2,'b-',label='weight focal global prior')
l1=plt.plot(momentum,cp3,'b--',label='weight focal iter prior')
l2=plt.plot(momentum,cp4,'-',color='#99aa55',label='baseline weight focal global prior')
l1=plt.plot(momentum,cp5,'--',color='#99aa55',label='baseline weight focal iter prior')
# l2=plt.plot(momentum,cp6,'--',color='#ff9955',label='focal loss')
# l1=plt.plot(momentum,cp7,'-',color='#55aa99',label='weight dicev2 loss')
# l2=plt.plot(momentum,cp8,'--',color='#55aa99',label='dicev2 loss')
plt.title('mIOU of weight-focal loss var momentum')
plt.xlabel('momentum')
plt.ylabel('miou')
plt.legend()
plt.show()
