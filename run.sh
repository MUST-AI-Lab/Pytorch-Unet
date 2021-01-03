python3 trainv3.py  --experiment keyboard_320 --dataset KeyBoard --data_dir  ./data/keyboard_320_320  --input_channels 3  --num_classes 2 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true --device cuda

python3 trainv3.py --load ./checkpoint/CP_ex.keyboard_320_epoch30_UNet_KeyBoard.pth --no_replace 'data_dir,experiment' --experiment keyboard_480  --data_dir  ./data/keyboard_480_480

python3 trainv3.py --load ./checkpoint/CP_ex.keyboard_480_epoch30_UNet_KeyBoard.pth --no_replace 'data_dir,experiment' --experiment keyboard_600  --data_dir  ./data/keyboard_600_600

python3 trainv3.py --load ./checkpoint/CP_ex.keyboard_600_epoch30_UNet_KeyBoard.pth --no_replace 'data_dir,experiment' --experiment keyboard_720  --data_dir  ./data/keyboard_720_720
