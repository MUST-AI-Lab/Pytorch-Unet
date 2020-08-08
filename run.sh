python3 train_dsb.py --exprtiment DSB00001 --batch-size 10 --arch UNet --deep_supervision false --optimizer SGD 

python3 train_dsb.py --exprtiment DSB00002 --batch-size 10 --arch NestedUNet --deep_supervision false --optimizer SGD

python3 train_dsb.py --exprtiment DSB00003 --batch-size 10 --arch NestedUNet --deep_supervision true --optimizer SGD

python3 train_mic.py --exprtiment MIC00001 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1

python3 train_dsb.py --exprtiment MIC00002 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1

python3 train_dsb.py --exprtiment MIC00003 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1

python3 train_dsb.py --exprtiment MIC00004 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false

python3 train_dsb.py --exprtiment MIC00005 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1

python3 train_dsb.py --exprtiment MIC00006 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false






