import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.tools import file_name

dir_name = ("./result/keyboard")
#dir_name = ("./result/trace")
x_axis=range(1,31)
file_array = file_name(dir_name)
res = {}
res['model_name']=[]
res['total_fn']=[]
res['true_anormaly_part']=[]
res['total_anormaly_part']=[]
res['error_predict_part']=[]
res['Anormaly_precision']=[]
res['Anormaly_recall']=[]
res['Anormaly_iou']=[]
res['Anormaly_tp']=[]
res['Anormaly_fp']=[]
res['Anormaly_fn']=[]


for file_name in file_array:
    res['model_name'].append(file_name[:-4])
    csv_data = pd.read_csv("{}/{}".format(dir_name,file_name))
    fn = np.array(csv_data["fn_1"])
    tp = np.array(csv_data["tp_1"])
    tn = np.array(csv_data["tn_1"])
    fp = np.array(csv_data["fp_1"])
    #total fn
    res['total_fn'].append(np.sum(fn))
    #error predict part
    tp_t = (tp>0).astype(np.int)
    fp_t = (fp>0).astype(np.int)
    uni = ((tp_t+fp_t)>0).astype(np.int)
    total_anormaly_count = np.sum(uni)
    total_false_count =  np.sum(fn>0)
    res['true_anormaly_part'].append(np.sum(tp_t))
    res['total_anormaly_part'].append(total_anormaly_count)
    res['error_predict_part'].append(total_false_count-total_anormaly_count)
    #Anormaly_precision
    Anormaly_tp = np.sum(tp*uni)*1.0
    Anormaly_fp = np.sum(fp*uni)*1.0
    Anormaly_fn = np.sum(fn*uni)*1.0
    res['Anormaly_tp'].append(Anormaly_tp)
    res['Anormaly_fp'].append(Anormaly_fp)
    res['Anormaly_fn'].append(Anormaly_fn)
    res['Anormaly_precision'].append(Anormaly_tp/(Anormaly_tp+Anormaly_fp))
    res['Anormaly_recall'].append(Anormaly_tp/(Anormaly_tp+Anormaly_fn))
    res['Anormaly_iou'].append(Anormaly_tp/(Anormaly_tp+Anormaly_fp+Anormaly_fn))

df = pd.DataFrame(res)
df.to_csv("./results_summary/keyboard_total_check.csv",index=None)




