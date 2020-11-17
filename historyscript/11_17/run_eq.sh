#1
python3 train.py  --experiment baseline_eq_b1_m070_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m080_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m090_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m091_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m092_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m093_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 4


#2
python3 train.py  --experiment baseline_eq_b1_m070_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m080_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m092_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distribute_weight --device_id 4

python3 train.py  --experiment baseline_eq_b1_m093_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss EqualizationLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distribute_weight --device_id 4




