python3 trainv3_recall.py  --experiment ce_wrecall --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3_recall.py  --experiment focal_wrecall --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss MultiFocalLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3_wmiou2_orign.py  --experiment ce_wmiou2_orign --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3_wmiou2_orign.py  --experiment focal_wmiou2_orign --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss MultiFocalLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3_wmiou2.py  --experiment ce_wmiou2 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3_wmiou2.py  --experiment focal_wmiou2 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss MultiFocalLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true




