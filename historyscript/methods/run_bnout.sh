#TDE
#ce
python3 train_tde.py  --experiment ce_tde_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetTDEBNout --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true

python3 train_tde.py  --experiment wce_tde_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetTDEBNout --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment wce_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetBnout --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment focal_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetBnout --deep_supervision false --optimizer SGD  --loss MultiFocalLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment wfocal_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetBnout --deep_supervision false --optimizer SGD  --loss MultiFocalLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true

python3 trainv3.py  --experiment EqualizationLossV3_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetBnout --deep_supervision false --optimizer SGD  --loss EqualizationLossV3  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_distribute_weight --device_id 0 --loss_reduce true --tail_radio 0.05

python3 trainv3.py  --experiment EqualizationLossV4_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetBnout --deep_supervision false --optimizer SGD  --loss EqualizationLossV4  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_distribute_weight --device_id 0 --loss_reduce true --tail_radio 0.05

python3 trainv3.py  --experiment EqualizationLossV3_Float_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetBnout --deep_supervision false --optimizer SGD  --loss EqualizationLossV3_Float  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_distribute_weight --device_id 0 --loss_reduce true 

python3 trainv3.py  --experiment EqualizationLossV4_Float_bnout --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetBnout --deep_supervision false --optimizer SGD  --loss EqualizationLossV4_Float  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_distribute_weight --device_id 0 --loss_reduce true 

