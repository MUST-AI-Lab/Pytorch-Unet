python3 trainv3.py  --experiment ce_b1_m070 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 trainv3.py  --experiment baseline_wce_b1_m070_single --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m070_batch --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m070_global --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 0

python3 trainv3.py  --experiment ce_b1_m080 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 trainv3.py  --experiment baseline_wce_b1_m080_single --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m080_batch --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m080_global --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 0

python3 trainv3.py  --experiment ce_b1_m090 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m090_single --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m090_batch --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m090_global --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 0

python3 trainv3.py  --experiment ce_b1_m091 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 trainv3.py  --experiment baseline_wce_b1_m091_single --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m091_batch --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m091_global --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 0


python3 trainv3.py  --experiment ce_b1_m092 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 trainv3.py  --experiment baseline_wce_b1_m092_single --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m092_batch --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m092_global --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 0

python3 trainv3.py  --experiment ce_b1_m093 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0

python3 trainv3.py  --experiment baseline_wce_b1_m093_single --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m093_batch --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0
python3 trainv3.py  --experiment baseline_wce_b1_m093_global --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight --device_id 0




