# switch to nnUNet

## 2020-08-13
This file record the experiment result of standard nnUnet learning process. Meanwhile, comparing the results with experiments in init_search.md



### HeLa data set
**baselise**
pic_size=512*512
| ID      | methods    | optimizer | lr  | loss |momentum| Dice(max)    |Dice(last)|modelsize(M)|
| ------- | ---------- | --------- | --- | ----- | --- |------ |--|---|
| HeLa0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 | 0.9260|0.9242|36.7
| HeLa0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 | 0.9310|0.9213|36.7
| HeLa0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.9191|0.8802|31.5
-|nnUNet  |SGD|1e-2|BCEDiceLoss|0.99|0.9389|0.9353|<165

### ISBI data set
**baselise**
pic_size=512*512
| ID      | methods    | optimizer | lr  | loss |momentum| Dice(max)    |Dice(last)  | modelsize(M)|
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- |--|
| ISBI0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |0.9499|0.9351|36.7
| ISBI0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.9481|0.9422|36.7
| ISBI0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.9439|0.9421|31.5
|-|nnUNet  |SGD|1e-2|BCEDiceLoss|0.99|0.8295|0.8291|<165|

### DSB data set
**baselise**
pic_size=96*96
| ID      | methods    | optimizer | lr  | loss |momentum| Dice (max)   |Dice(last) |model_size|
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- |--|
| DSB0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 | 0.9228|0.9224|36.7|
| DSB0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.9225|0.9223|36.7|
| DSB0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.9015|0.8809|31.5|
|-|nnUNet  |SGD|1e-2|BCEDiceLoss|0.99|0.9259|0.9110|<28|
