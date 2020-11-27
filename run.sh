python3 trainv3.py  --experiment trace_detail_eq --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss EqualizationLossV2  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment trace_detail_filter --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss FilterCELossV2  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --device_id 0 --loss_reduce true



#------------------------filter loss---------------------------
python3 trainv3.py  --experiment filter_tailredio1.0 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 1.0  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0.5 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.5  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0.28 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.28  --device_id 0 --loss_reduce true

python3 train.py  --experiment filter_tailredio0.26 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.26  --device_id 0 --loss_reduce true


python3 trainv3.py  --experiment filter_tailredio0.17 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.17  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0.1 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.1  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0.05 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.05  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0.03 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.03  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0.02 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.02  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0.01 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0.01  --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment filter_tailredio0 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss FilterCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distribute_weight --tail_radio 0  --device_id 0 --loss_reduce true
