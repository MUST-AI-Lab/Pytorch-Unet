python3 train.py  --experiment MultiFocalLoss --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 train.py  --experiment MultiFocalLossV3 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 train.py  --experiment MultiFocalLossV3 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 0

python3 train.py  --experiment MultiFocalLoss_0 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.0 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 train.py  --experiment MultiFocalLossV3_0 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.0 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 train.py  --experiment MultiFocalLossV3_0 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.0 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 0


