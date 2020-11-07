python3 train.py  --experiment ce_batch1 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch FCNNhub --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none

python3 train.py  --experiment ce_batch1_single_test_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch FCNNhub --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_test_weight

python3 train.py  --experiment ce_batch1_batch_test_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch FCNNhub --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_test_weight

python3 train.py  --experiment ce_batch1_global_test_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch FCNNhub --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_test_weight

python3 train.py  --experiment ce_batch1_single_baseline_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch FCNNhub --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight

python3 train.py  --experiment ce_batch1_batch_baseline_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch FCNNhub --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight

python3 train.py  --experiment ce_batch1_global_global_baseline_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch FCNNhub --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight

#=========================================================
python3 train.py  --experiment ce_batch2 --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch FCNNhub --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none

python3 train.py  --experiment ce_batch2_single_test_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch FCNNhub --deep_supervision false --optimizer SGD  --weight_loss false  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_test_weight

python3 train.py  --experiment ce_batch2_batch_test_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch FCNNhub --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_test_weight

python3 train.py  --experiment ce_batch2_global_test_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch FCNNhub --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_test_weight

python3 train.py  --experiment ce_batch2_single_baseline_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch FCNNhub --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_baseline_weight

python3 train.py  --experiment ce_batch2_batch_baseline_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch FCNNhub --deep_supervision false --optimizer SGD    --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight

python3 train.py  --experiment ce_batch2_global_global_baseline_weight --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch FCNNhub --deep_supervision false --optimizer SGD   --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_baseline_weight

