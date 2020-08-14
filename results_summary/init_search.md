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
| BASIC0013 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0014 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0015 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0016 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0017 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0018 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0019 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0020 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0021 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0022 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0023 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0024 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0025 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0026 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0027 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0028 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0029 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0030 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0031 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0032 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0033 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0034 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0035 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0036 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

### U373 data set
#### Unet
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0037 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0038 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0039 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0040 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0041 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0042 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0043 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0044 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0045 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0046 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0047 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0048 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0049 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0050 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0051 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0052 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0053 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0054 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0055 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0056 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0057 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0058 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0059 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0060 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

### HeLa data set
#### Unet
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0061 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0062 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0063 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0064 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0065 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0066 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0067 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0068 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0069 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0070 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0071 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0072 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0073 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0074 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0075 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0076 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0077 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0078 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0079 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0080 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0081 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0082 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0083 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0084 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

### ISBI data set
#### Unet
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0085 | UNet       | Adam      | 3e-4 | false   | 0.99 |
| BASIC0086 | UNet       | Adam       | 3e-3 | false   | 0.99 |
| BASIC0087 | UNet       | Adam       | 3e-2 | false   | 0.99 |
| BASIC0088 | UNet       | Adam       | 3e-1 | false   | 0.99 |
| BASIC0089 | UNet       | SGD       | 1e-1 | false   | 0.99 |
| BASIC0090 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0091 | UNet       | SGD       | 1e-3 | false   | 0.99 |
| BASIC0092 | UNet       | SGD       | 1e-4 | false   | 0.99 |

#### Unet++
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0093 | NestedUNet      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0094 | NestedUNet       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0095 | NestedUNet       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0096 | NestedUNet       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0097 | NestedUNet       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0098 | NestedUNet       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0099 | NestedUNet       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0100 | NestedUNet       | SGD       | 1e-4 | false   | 0.9 |

#### Unet++(with super vision)
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0101 | NestedUNet(sp)      | Adam      | 3e-4 | false   | 0.9 |
| BASIC0102 | NestedUNet(sp)       | Adam       | 3e-3 | false   | 0.9 |
| BASIC0103 | NestedUNet(sp)       | Adam       | 3e-2 | false   | 0.9 |
| BASIC0104 | NestedUNet(sp)       | Adam       | 3e-1 | false   | 0.9 |
| BASIC0105 | NestedUNet(sp)       | SGD       | 1e-1 | false   | 0.9 |
| BASIC0106 | NestedUNet(sp)       | SGD       | 1e-2 | false   | 0.9 |
| BASIC0107 | NestedUNet(sp)       | SGD       | 1e-3 | false   | 0.9 |
| BASIC0108 | NestedUNet(sp)       | SGD       | 1e-4 | false   | 0.9 |

## 3. focuse on weight loss  for different data set
using the default value only different in using weight loss or not
### DSB data set
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0109 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0110 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0111 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0112 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0113 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0114 | UNet       | SGD       | 1e-2 | true   | 0.99 |


### U373 dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0115 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0116 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0117 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0118 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0119 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0120 | UNet       | SGD       | 1e-2 | true   | 0.99 |

### HeLa dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0121 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0122 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0123 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0124 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0125 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0126 | UNet       | SGD       | 1e-2 | true   | 0.99 |

### ISBI dataset
| ID      | methods    | optimizer | lr  | weight_loss |momentum| IOU    |
| ------- | ---------- | --------- | --- | ----- | --- |------ |
| BASIC0126 | NestedUNet       | Adam      | 3e-4 | false   | 0.9 |
| BASIC0127 | NestedUNet       | Adam      | 3e-4 | true   | 0.9 |
| BASIC0128 | NestedUNet(sp)       | Adam       | 3e-4 | false   | 0.9 |
| BASIC0129 | NestedUNet(sp)       | Adam       | 3e-4 | true   | 0.9 |
| BASIC0130 | UNet       | SGD       | 1e-2 | false   | 0.99 |
| BASIC0131 | UNet       | SGD       | 1e-2 | true   | 0.99 |