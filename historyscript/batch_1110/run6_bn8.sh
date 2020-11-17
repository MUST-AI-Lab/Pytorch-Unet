
python3 train.py  --experiment ce_b8_m090 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 6

python3 train.py  --experiment test01_wce_b8_m090_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4--accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_test_weight --device_id 6
python3 train.py  --experiment test01_wce_b8_m090_test_batch --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_test_weight --device_id 6
python3 train.py  --experiment test01_wce_b8_m090_test_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_test_weight --device_id 6
python3 train.py  --experiment baseline_wce_b8_m090_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 6
python3 train.py  --experiment baseline_wce_b8_m090_batch --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 6
python3 train.py  --experiment baseline_wce_b8_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 6

python3 train.py  --experiment ce_b8_m091 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 6

python3 train.py  --experiment test01_wce_b8_m091_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32  --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_test_weight --device_id 6
python3 train.py  --experiment test01_wce_b8_m091_test_batch --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_test_weight --device_id 6
python3 train.py  --experiment test01_wce_b8_m091_test_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_test_weight --device_id 6
python3 train.py  --experiment baseline_wce_b8_m091_single --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 6
python3 train.py  --experiment baseline_wce_b8_m091_batch --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 6
python3 train.py  --experiment baseline_wce_b8_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --accumulation-step 2 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 6




