#test dataset ISBI
python3 train_mic.py --experiment MIC00012 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --data_name ISBI
python3 train_mic.py --experiment MIC00013 --batch-size 1 --arch NestedUNet --deep_supervision false --optimizer SGD --scale -1 --data_name ISBI
python3 train_mic.py --experiment MIC00014 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --weight_loss false --data_name ISBI
python3 train_mic.py --experiment MIC00015 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD --scale -1 --data_name ISBI
python3 train_mic.py --experiment MIC00016 --batch-size 1 --arch NestedUNet --deep_supervision true --optimizer SGD --scale -1 --weight_loss false --data_name ISBI