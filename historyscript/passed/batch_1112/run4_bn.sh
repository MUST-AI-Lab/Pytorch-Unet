#1
python3 train.py  --experiment baseline_wce_b1_m070_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b1_m080_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b1_m090_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b1_m091_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b1_m092_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b1_m093_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

#2
python3 train.py  --experiment baseline_wce_b2_m070_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b2_m080_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b2_m090_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b2_m091_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b2_m092_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b2_m093_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

#4
python3 train.py  --experiment baseline_wce_b4_m070_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b4_m080_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b4_m090_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b4_m091_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b4_m092_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b4_m093_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4


#8
python3 train.py  --experiment baseline_wce_b8_m070_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b8_m080_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b8_m090_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b8_m091_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b8_m092_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

python3 train.py  --experiment baseline_wce_b8_m093_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 4

