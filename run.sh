python3 train.py  --experiment CAM_weight_ce --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --weight_loss false  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.98 --save_check_point false  --force_save_last true  --weight_loss True --weight_type pixel

python3 train.py  --experiment CAM_ce --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --weight_loss false  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.98 --save_check_point false  --force_save_last true --weight_loss False

python3 train.py  --experiment CAM_weight_focal --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --weight_loss false  --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.98 --save_check_point false  --force_save_last true  --weight_loss True --weight_type pixel

python3 train.py  --experiment CAM_focal --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --weight_loss false  --loss MultiFocalLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.98 --save_check_point false  --force_save_last true --weight_loss False

python3 train.py  --experiment CAM_weight_dice --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --weight_loss false  --loss SoftDiceLossV2  --epochs 30 --learning-rate 1e-2  --momentum 0.98 --save_check_point false  --force_save_last true  --weight_loss True --weight_type distrubution

python3 train.py  --experiment CAM_dice --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --weight_loss false  --loss SoftDiceLossV2  --epochs 30 --learning-rate 1e-2  --momentum 0.98 --save_check_point false  --force_save_last true --weight_loss False

shutdown
