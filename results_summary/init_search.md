# initial  experiments
## 2020-08-13
We take the experiment as a searching process. the goal is to find out which hyper-parameter sub set of hyper-parameter space can cause better result of certain  situtation.
we should begin with a simple condition

the seatching hyper-parameter space is :
methods:{UNet,NestedUNet,NestedUNet(sp)}
optimizer:{SGD,Adam}
lr:{real number value [0,1]}
momentum:{real number value [0,1]}
weight_loss:{WeightBCELossNormal,WeightBCELoss,BCEWithLogitsLoss}
## reference default
### Unet
batch_size=1, optimizer=SGD,lr=0.01,momentum=0.99,weight_loss=true, loss=WeightBCELoss

### Unet++
batch_size=1, optimizer=adam,lr=3e-4,momentum=0.9,weight_loss=false,loss=BCEDiceLoss

**summary**
The loss function in Unet in 2015 weight loss by default, but there are no loss function  in NestedUNet 2018. 
The loss function of UNet 2015 is cross entropy loss  function with weight
The loss function of NestedUNet 2018 is a combination of binary cross-entropy and dice coefficient as the loss function
this is the **baseline** of experiment

### HeLa data set
**baselise**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| HeLa0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 | 0.8623
| HeLa0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 | 0.8709
| HeLa0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8504
**The Control group with weight**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| HeLa0004 | NestedUNet       | Adam      | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8647
| HeLa0005 | NestedUNet       | Adam      | 3e-4 | WeightBCELoss   | 0.9 |0.8671
| HeLa0006 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8644
| HeLa0007 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCELoss   | 0.9 |0.8632
| HeLa0008| UNet       | SGD       | 1e-2 | BCEWithLogitsLoss   | 0.99 |0.8304
| HeLa0009 | UNet       | SGD       | 1e-2 | WeightBCEDiceLoss   | 0.99 |0.8414
**The Control group with optimizer**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| HeLa0010 | UNet       | Adam       | 3e-4 | WeightBCELoss   | 0.99 |0.8419

**Test for new method:PyramidUNet**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| P_HeLa0003 | PyramidUNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8719
***
### ISBI data set
**baselise**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| ISBI0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |0.9048
| ISBI0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.9014
| ISBI0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8938
**The Control group with weight**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| ISBI0004 | NestedUNet       | Adam      | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8987
| ISBI0005 | NestedUNet       | Adam      | 3e-4 | WeightBCELoss   | 0.9 |0.8938
| ISBI0006 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8999
| ISBI0007 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCELoss   | 0.9 |0.8956
| ISBI0008| UNet       | SGD       | 1e-2 | BCEWithLogitsLoss   | 0.99 |0.8995
| ISBI0009 | UNet       | SGD       | 1e-2 | WeightBCEDiceLoss   | 0.99 |0.9023
**The Control group with optimizer**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| ISBI0010 | UNet       | Adam       | 3e-4 | WeightBCELoss   | 0.99 |0.8595

**Test for new method:PyramidUNet**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| ISBI0003 | UNet       | PyramidUNet       | 1e-2 | WeightBCELoss   | 0.99 |
***
### DSB data set
**baselise**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| DSB0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |
| DSB0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |
| ISBI0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |
**The Control group with weight**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| DSB0004| NestedUNet       | Adam      | 3e-4 | WeightBCEDiceLoss   | 0.9 |
| DSB0005 | NestedUNet       | Adam      | 3e-4 | WeightBCELoss   | 0.9 |
| DSB0006 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCEDiceLoss   | 0.9 |
| DSB0007 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCELoss   | 0.9 |
| DSB0008| UNet       | SGD       | 1e-2 | BCEWithLogitsLoss   | 0.99 |
| DSB0009 | UNet       | SGD       | 1e-2 | WeightBCEDiceLoss   | 0.99 |
**The Control group with optimizer**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| DSB0010 | UNet       | Adam       | 3e-4 | WeightBCELoss   | 0.99 |

**Test for new method:PyramidUNet**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| ISBI0003 | PyramidUNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |

***



