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
| ID      | methods    | optimizer | lr  | loss |momentum| IOU(max)    |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--|
| HeLa0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 | 0.8623|0.8591
| HeLa0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 | 0.8709|0.8541
| HeLa0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8504|0.7861
**The Control group with weight**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU(max)    |IOU(last)    |
| ------- | ---------- | --------- | --- | --- | ----- | --- |------ |
| HeLa0004 | NestedUNet       | Adam      | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8647|0.8516
| HeLa0005 | NestedUNet       | Adam      | 3e-4 | WeightBCELoss   | 0.9 |0.8671|0.8173
| HeLa0006 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8644|0.8559
| HeLa0007 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCELoss   | 0.9 |0.8632|0.8367
| HeLa0008| UNet       | SGD       | 1e-2 | BCEWithLogitsLoss   | 0.99 |0.8304|0.8084
| HeLa0009 | UNet       | SGD       | 1e-2 | WeightBCEDiceLoss   | 0.99 |0.8414|0.8202
**The Control group with optimizer**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU(max)    |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| HeLa0010 | UNet       | Adam       | 3e-4 | WeightBCELoss   | 0.99 |0.8419|0.8180

**Test for new method:PyramidUNet**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU (max)   |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| P_HeLa0001 | PyramidNestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |0.8469|0.8275
| P_HeLa0002 | PyramidNestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.8644|0.8005
| P_HeLa0003 | PyramidUNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8743|0.8670

***
### ISBI data set
**baselise**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU(max)    |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| ISBI0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |0.9048|0.8786
| ISBI0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.9014|0.8910
| ISBI0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8938|0.8906
**The Control group with weight**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU   (max) |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| ISBI0004 | NestedUNet       | Adam      | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8987|0.8888
| ISBI0005 | NestedUNet       | Adam      | 3e-4 | WeightBCELoss   | 0.9 |0.8938|0.8711
| ISBI0006 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8999|0.8855
| ISBI0007 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCELoss   | 0.9 |0.8956|0.8793
| ISBI0008| UNet       | SGD       | 1e-2 | BCEWithLogitsLoss   | 0.99 |0.8995|0.8983
| ISBI0009 | UNet       | SGD       | 1e-2 | WeightBCEDiceLoss   | 0.99 |0.9023|0.9012
**The Control group with optimizer**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU (max)   |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| ISBI0010 | UNet       | Adam       | 3e-4 | WeightBCELoss   | 0.99 |0.8595|0.8396

**Test for new method:PyramidUNet**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU    |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| P_ISBI0001 | PyramidNestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |0.9108|0.9059
| P_ISBI0002 | PyramidNestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.8988|0.8904
| P_ISBI0003 | PyramidUNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8834|0.8791
***
### DSB data set
**baselise**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU (max)   |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| DSB0001 | NestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 | 0.8595|0.8589
| DSB0002 | NestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.8589|0.8587
| ISBI0003 | UNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8275|0.8082
**The Control group with weight**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU (max)   |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| DSB0004| NestedUNet       | Adam      | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8593|0.8562
| DSB0005 | NestedUNet       | Adam      | 3e-4 | WeightBCELoss   | 0.9 |0.7779|0.7337
| DSB0006 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCEDiceLoss   | 0.9 |0.8586|0.8030
| DSB0007 | NestedUNet(sp)       | Adam       | 3e-4 | WeightBCELoss   | 0.9 |0.8210|0.8571
| DSB0008| UNet       | SGD       | 1e-2 | BCEWithLogitsLoss   | 0.99 |0.8371|0.8302
| DSB0009 | UNet       | SGD       | 1e-2 | WeightBCEDiceLoss   | 0.99 |0.8449|0.8431
**The Control group with optimizer**
| ID      | methods    | optimizer | lr  | loss |momentum| IOU(max)    |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| DSB0010 | UNet       | Adam       | 3e-4 | WeightBCELoss   | 0.99 |0.8460|0.7930

| ID      | methods    | optimizer | lr  | loss |momentum| IOU (max)   |IOU(last)    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |--- | 
| P_DSB0001 | PyramidNestedUNet       | Adam      | 3e-4 | BCEDiceLoss   | 0.9 |0.8594|0.8579
| P_DSB0002 | PyramidNestedUNet(sp)       | Adam       | 3e-4 | BCEDiceLoss   | 0.9 |0.8593|0.8574
| P_DSB0003 | PyramidUNet       | SGD       | 1e-2 | WeightBCELoss   | 0.99 |0.8391|0.8360
***



