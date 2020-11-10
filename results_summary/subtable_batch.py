import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

subtablelist=['ce_b2','baseline_batch_b2','baseline_global_b2','baseline_single_b2','test_batch_b2','test_global_b2','test_single_b2']

csv_data = pd.read_csv("./results_summary/weight_momentum_batch.csv")
momentum=csv_data['momentum']

new_table = dict()
new_table['ex/mon'] = subtablelist
for item in momentum:
    new_table[item] = []

for item in subtablelist:
    iter =0
    for mon in momentum:
        new_table[mon].append(csv_data[item][iter])
        iter +=1

df = pd.DataFrame(new_table)
df.to_csv("./results_summary/ce_batch_2.csv",index=None)



