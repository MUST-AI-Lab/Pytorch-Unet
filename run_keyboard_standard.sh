python3 trainv3.py  --experiment keyboard_ce --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment keyboard_ce_iw --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment keyboard_focal --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss MultiFocalLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment keyboard_focal_iw --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss MultiFocalLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true


#Dice
python3 trainv3.py  --experiment keyboard_dice --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss DiceLossV3  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment keyboard_dice_iw --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss DiceLossV3  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

#ASL
python3 trainv3.py  --experiment keyboard_asl --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss ASLLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment keyboard_asl_iw --dataset KeyBoard2 --data_dir  ./data/dataset_4type_keyboard  --input_channels 3  --num_classes 4 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss ASLLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true
