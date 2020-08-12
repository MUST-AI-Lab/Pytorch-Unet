# A markdown for result log summary



## 2020-08-12

### 1. DSB DataSet Collection

| ID      | methods    | optimizer | lr  | epoch | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| DSB00001 | UNet       | SGD       | 0.1 | 100   | 0.8036 |
| DSB00007 | UNet       | Adam       | 0.1 | 100   | 0.7577 |
| DSB00010 | UNet       | Adam       | 0.01 | 100   | 0.8078 |
| DSB00004 | UNet       | Adam       | 0.001 | 100   | 0.8133 |

| ID      | methods    | optimizer | lr  | epoch | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| DSB00002 | UNet++     | SGD       | 0.1 | 100   | 0.7984 |
| DSB00008 | UNet++       | Adam       | 0.1 | 100   | 0.7955 |
| DSB00011 | UNet++       | Adam       | 0.01 | 100   | 0.8123 |
| DSB00005 | UNet++       | Adam       | 0.001 | 100   | 0.8173 |

(sp):means deepsupervision

| ID      | methods    | optimizer | lr  | epoch | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| DSB00003 | UNet++(sp) | SGD       | 0.1 | 100   | 0.8177 |
| DSB00009 | UNet++(sp)       | Adam       | 0.1 | 100   | 0.7781 |
| DSB00012 | UNet++(sp)       | Adam       | 0.01 | 100   | 0.8047 |
| DSB00006 | UNet++(sp)       | Adam       | 0.001 | 100   | 0.8127 |

```
first test
python3 train_dsb.py --experiment DSB00001 --batch-size 10 --arch UNet --deep_supervision false --optimizer SGD
python3 train_dsb.py --experiment DSB00002 --batch-size 10 --arch NestedUNet --deep_supervision false --optimizer SGD
python3 train_dsb.py --experiment DSB00003 --batch-size 10 --arch NestedUNet --deep_supervision true --optimizer SGD

#test  for adam
python3 train_dsb.py --experiment DSB00004 --batch-size 10 --arch UNet --deep_supervision false --optimizer Adam  --learning-rate 1e-3
python3 train_dsb.py --experiment DSB00005 --batch-size 10 --arch NestedUNet --deep_supervision false --optimizer Adam  --learning-rate 1e-3
python3 train_dsb.py --experiment DSB00006 --batch-size 10 --arch NestedUNet --deep_supervision true --optimizer Adam  --learning-rate 1e-3

python3 train_dsb.py --experiment DSB00007 --batch-size 10 --arch UNet --deep_supervision false --optimizer Adam  --learning-rate 1e-1
python3 train_dsb.py --experiment DSB00008 --batch-size 10 --arch NestedUNet --deep_supervision false --optimizer Adam  --learning-rate 1e-1
python3 train_dsb.py --experiment DSB00009 --batch-size 10 --arch NestedUNet --deep_supervision true --optimizer Adam  --learning-rate 1e-1

python3 train_dsb.py --experiment DSB00010 --batch-size 10 --arch UNet --deep_supervision false --optimizer Adam  --learning-rate 1e-2
python3 train_dsb.py --experiment DSB00011 --batch-size 10 --arch NestedUNet --deep_supervision false --optimizer Adam  --learning-rate 1e-2
python3 train_dsb.py --experiment DSB00012 --batch-size 10 --arch NestedUNet --deep_supervision true --optimizer Adam  --learning-rate 1e-2
```

### 2. MIC DataSet Collection



#### 2.1 U373 Data Set
all in epoch=100, but
there something worng in setting experiment ...

| ID      | methods    | optimizer | lr  | weight_loss | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| MIC00001 | NestedUNet       | SGD       | 0.1 | true   | 0.8584 |
| MIC00002 | NestedUNet(sp)       | SGD       | 0.1 | true   | - |
| MIC00003 | NestedUNet       | SGD       | 0.1 | true   | 0.8436 |
| MIC00006 | NestedUNet(sp)       | SGD       | 0.1 | false   | - |
| MIC00004 | UNet       | SGD       | 0.1 | false   | 0.8915 |
| MIC00005 | UNet       | SGD       | 0.1 | true   | 0.8540 |


```
first test for deep supervision and  weight loss
#test dataset U373
python3 train_mic.py --experiment MIC00001 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1
python3 train_mic.py --experiment MIC00002 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1
python3 train_mic.py --experiment MIC00003 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1
python3 train_mic.py --experiment MIC00004 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false
python3 train_mic.py --experiment MIC00005 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1
python3 train_mic.py --experiment MIC00006 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false
```

#### 2.2 HeLa Data Set
| ID      | methods    | optimizer | lr  | weight_loss | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| MIC00007 | NestedUNet(sp)       | SGD       | 0.1 | true   | 0.8530 |
| MIC00011 | NestedUNet(sp)       | SGD       | 0.1 | false   | 0.8475 |
| MIC00008 | NestedUNet       | SGD       | 0.1 | true   | 0.8156 |
| MIC00010 | UNet       | SGD       | 0.1 | true   | 0.8556 |
| MIC00009 | UNet       | SGD       | 0.1 | false   | 0.7764 |


```
#test dataset HeLa
python3 train_mic.py --experiment MIC00007 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --data_name Hela
python3 train_mic.py --experiment MIC00008 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --data_name Hela
python3 train_mic.py --experiment MIC00009 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false --data_name Hela
python3 train_mic.py --experiment MIC00010 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --data_name Hela
python3 train_mic.py --experiment MIC00011 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false --data_name Hela
```

#####  Weight Loss collection (for HeLa)
| ID      | methods    | optimizer | lr  | weight_loss | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| MIC00017 | NestedUNet       | SGD       | 0.1 | BCEWithLogitsLoss   | 0.8368 |
| MIC00018 | NestedUNet       | SGD       | 0.1 | WeightBCELoss   | 0.8842 |
| MIC00019 | NestedUNet       | SGD       | 0.1 | WeightBCELossNormal   | 0.8217 |

| ID      | methods    | optimizer | lr  | weight_loss | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| MIC00020 | NestedUNet(sp)        | SGD       | 0.1 | BCEWithLogitsLoss   | 0.8304 |
| MIC00021 | NestedUNet(sp)        | SGD       | 0.1 | WeightBCELoss   | 0.8549 |
| MIC00022 | NestedUNet(sp)        | SGD       | 0.1 | WeightBCELossNormal   | 0.8459 |

| ID      | methods    | optimizer | lr  | weight_loss | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| MIC00036 | UNet       | SGD       | 0.1 | BCEWithLogitsLoss   | 0.8460 |
| MIC00037 | UNet       | SGD       | 0.1 | WeightBCELoss   | 0.8715 |
| MIC00038 | UNet       | SGD       | 0.1 | WeightBCELossNormal   | 0.8395 |

```
#for NestedUNet  without deep_supervision
python3 train_mic.py --experiment MIC00017 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false  --loss BCEWithLogitsLoss --data_name Hela --epochs 60
python3 train_mic.py --experiment MIC00018 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss true  --loss WeightBCELoss --data_name Hela --epochs 60
python3 train_mic.py --experiment MIC00019 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss true  --loss WeightBCELossNormal --data_name Hela --epochs 60

#for NestedUNet  with deep_supervision
python3 train_mic.py --experiment MIC00020 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false  --loss BCEWithLogitsLoss --data_name Hela --epochs 60
python3 train_mic.py --experiment MIC00021 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss true  --loss WeightBCELoss --data_name Hela --epochs 60
python3 train_mic.py --experiment MIC00022 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss true  --loss WeightBCELossNormal --data_name Hela --epochs 60

#for UNet  has no deep_supervision implment
python3 train_mic.py --experiment MIC00036 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false  --loss BCEWithLogitsLoss --data_name HeLa --epochs 60
python3 train_mic.py --experiment MIC00037 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss true  --loss WeightBCELoss --data_name HeLa --epochs 60
python3 train_mic.py --experiment MIC00038 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss true  --loss WeightBCELossNormal --data_name HeLa --epochs 60
```



#### 2.3 ISBI Data Set
| ID      | methods    | optimizer | lr  | weight_loss | IOU    |
| ------- | ---------- | --------- | --- | ----- | ------ |
| MIC00012 | NestedUNet(sp)       | SGD       | 0.1 | true   | 0.8636 |
| MIC00016 | NestedUNet(sp)       | SGD       | 0.1 | false   | 0.9043 |
| MIC00013 | NestedUNet       | SGD       | 0.1 | true   | 0.7933 |
| MIC00015 | UNet       | SGD       | 0.1 | true   | 0.8938 |
| MIC00014 | UNet       | SGD       | 0.1 | false   | 0.8149 |

```

test dataset ISBI
python3 train_mic.py --experiment MIC00012 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --data_name ISBI
python3 train_mic.py --experiment MIC00013 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --data_name ISBI
python3 train_mic.py --experiment MIC00014 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false --data_name ISBI
python3 train_mic.py --experiment MIC00015 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --data_name ISBI
python3 train_mic.py --experiment MIC00016 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false --data_name ISBI
```
