import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

subtablelist=['ce','ce_baseline_global','ce_baseline','ce_weight','ce_weight_local']

csv_data = pd.read_csv("./results_summary/weight_momentum.csv")
momentum=csv_data['mon']

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
df.to_csv("./results_summary/home_ce.csv",index=None)



