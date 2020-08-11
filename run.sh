# first test
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


# first test for deep supervision and  weight loss
python3 train_mic.py --experiment MIC00001 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1

python3 train_mic.py --experiment MIC00002 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1

python3 train_mic.py --experiment MIC00003 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1

python3 train_mic.py --experiment MIC00004 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false

python3 train_mic.py --experiment MIC00005 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1

python3 train_mic.py --experiment MIC00006 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false

python3 train_mic.py --experiment MIC00007 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --data_name Hela

python3 train_mic.py --experiment MIC00008 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --data_name Hela

python3 train_mic.py --experiment MIC00009 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false --data_name Hela

python3 train_mic.py --experiment MIC00010 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --data_name Hela

python3 train_mic.py --experiment MIC00011 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false --data_name Hela

python3 train_mic.py --experiment MIC00012 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --data_name ISBI

python3 train_mic.py --experiment MIC00013 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --data_name ISBI

python3 train_mic.py --experiment MIC00014 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false --data_name ISBI

python3 train_mic.py --experiment MIC00015 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --data_name ISBI

python3 train_mic.py --experiment MIC00016 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false --data_name ISBI


#test for different implement  weights for loss





