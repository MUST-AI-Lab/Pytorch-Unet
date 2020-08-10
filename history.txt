python3 train_dsb.py --experiment DSB00001 --batch-size 10 --arch UNet --deep_supervision false --optimizer SGD 

python3 train_dsb.py --experiment DSB00002 --batch-size 10 --arch NestedUNet --deep_supervision false --optimizer SGD

python3 train_dsb.py --experiment DSB00003 --batch-size 10 --arch NestedUNet --deep_supervision true --optimizer SGD

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






