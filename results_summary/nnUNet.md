# switch to nnUNet

## 2020-09-01
This file record the experiment result of standard nnUnet learning process. Meanwhile, comparing the results with experiments in init_search.md



### HeLa data set
**baselise**
pic_size=512*512
| ID      | methods    | optimizer | lr  | loss |momentum| Dice(max)    |Dice(last)|modelsize(M)|
| ------- | ---------- | --------- | --- | ----- | --- |------ |--|---|
| HeLa0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 | 0.9260|0.9242|36.7
| HeLa0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 | 0.9310|0.9213|36.7
| HeLa0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.9191|0.8802|31.5
-|nnUNet  |SGD|1e-2|BCEDiceLoss|0.99|0.9379|0.9365|<165

**net-arch(Encoder only, decoder is the mirror of encoder), the upsampling method is ConvTranspose2d**
|layers| arch      | detail    | 
|--| ------- | ---------- |
|1|Conv2d|(1, 32, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|2|Conv2d|(32, 64, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|3|Conv2d|(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|4|Conv2d|(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|5|Conv2d|(256, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(480, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|6|Conv2d|(480, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(480, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|7|Conv2d|(480, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(480, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|


### ISBI data set
**baselise**
pic_size=512*512
| ID      | methods    | optimizer | lr  | loss |momentum| Dice(max)    |Dice(last)  | modelsize(M)|
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- |--|
| ISBI0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |0.9499|0.9351|36.7
| ISBI0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.9481|0.9422|36.7
| ISBI0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.9439|0.9421|31.5
|-|nnUNet  |SGD|1e-2|BCEDiceLoss|0.99|0.8268|0.8227|<165|

**net-arch same as HeLa**

### DSB data set
**baselise**
pic_size=96*96
| ID      | methods    | optimizer | lr  | loss |momentum| Dice (max)   |Dice(last) |model_size|
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- |--|
| DSB0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 | 0.9228|0.9224|36.7|
| DSB0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.9225|0.9223|36.7|
| DSB0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.9015|0.8809|31.5|
|-|nnUNet  |SGD|1e-2|BCEDiceLoss|0.99|0.9329|0.9236|<28|

**net-arch(Encoder only, decoder is the mirror of encoder), the upsampling method is ConvTranspose2d**
|layers| arch      | detail    | 
|--| ------- | ---------- |
|1|Conv2d|(3, 32, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|2|Conv2d|(32, 64, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|3|Conv2d|(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|4|Conv2d|(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
|5|Conv2d|(256, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|
||Conv2d|(480, 480, kernel_size=[3, 3], stride=(1, 1), padding=[1, 1])|


## 2020-09-12
这个星期的实验围绕最小的特征图大小展开：主要的实验设置是设置最小的特征图大小为4，8，16，32，64。实验针对三个数据集单独设置不同的数据集横向比对。
实验的载体均为nnUnet, 优化器为SGD，初始学习率为1e-2, 损失函数设置为BCEDiceLoss，动量0.99, 考虑HeLa数据集参与训练的只有7张，所有实验统一设置barch size为1。 nnUnet每个epoch固定250次迭代，每个实验运行50个epoch。

### HeLa data set
| ID |  Minimum feature map     |  Dice(max)    |Dice(last)|modelsize(M):total|
| ------- |--|---|---|---|
|Task001_HeLa|4|0.8352|0.7532|425|
|Task002_HeLa|8|0.9379|0.9365|330|
|Task004_HeLa|16|0.8238|0.8223|239|
|Task006_HeLa|32|0.9401|0.9393|149|
|Task007_HeLa|64|0.8404|0.8086|59|

### ISBI data set
| ID |  Minimum feature map     |  Dice(max)    |Dice(last)|modelsize(M):total|
| ------- |--|---|---|---|
|Task089_ISBI|4|0.8286|0.8261|425|
|Task090_ISBI|8|0.8268|0.8227|330|
|Task092_ISBI|16|0.8238|0.8196|239|
|Task094_ISBI|32|0.8248|0.8164|149|
|Task095_ISBI|64|0.8279|0.8199|59|

### DSB data set
| ID |  Minimum feature map     |  Dice(max)    |Dice(last)|modelsize(M):total|
| ------- |--|---|---|---|
|Task011_DSB|4|0.9126|0.9117|149|
|Task012_DSB|8|0.9329|0.9236|59|
|Task014_DSB|16|0.9222|0.9072|15.5|
|Task016_DSB|32|0.9238|0.9145|3.8|
|Task017_DSB|64|0.9300|0.9187|0.8|

**detail  net arch in arch-nnUnet.xslx** 