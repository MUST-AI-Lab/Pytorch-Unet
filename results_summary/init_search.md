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
batch_size=1, optimizer=SGD,lr=0.01,momentum=0.99,weight_loss=true

### Unet++
batch_size=1, optimizer=adam,lr=3e-4,momentum=0.9,weight_loss=false

## 1. focus on methods with default value
### DSB data set
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0003 | UNet       | SGD       | 1e-2 | true   | 0.99 |


### U373 dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0004 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0005 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0006 | UNet       | SGD       | 1e-2 | true   | 0.99 |

### HeLa dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0007 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0008 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0009 | UNet       | SGD       | 1e-2 | true   | 0.99 |

### ISBI dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0012 | UNet       | SGD       | 1e-2 | true   | 0.99 |

## 2. focus on optimizer  and learning rate
### DSB data set
#### Unet
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

### U373 data set
#### Unet
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

### HeLa data set
#### Unet
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

### ISBI data set
#### Unet
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0011 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0012 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0010 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0011 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0012 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

## 3. focuse on weight loss  for different data set
using the default value only different in using weight loss or not
### DSB data set
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0003 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0003 | UNet       | SGD       | 1e-2 | true   | 0.99 |


### U373 dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0003 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0003 | UNet       | SGD       | 1e-2 | true   | 0.99 |

### HeLa dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0003 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0003 | UNet       | SGD       | 1e-2 | true   | 0.99 |

### ISBI dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0001 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0002 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0003 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0003 | UNet       | SGD       | 1e-2 | true   | 0.99 |