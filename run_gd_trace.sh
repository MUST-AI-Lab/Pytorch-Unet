#CamVid
python3 main.py  --experiment ce --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true --device cuda --trainer GradientTraceTrainer

python3 main.py  --experiment ce_iw --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type batch_baseline_weight --device_id 0 --loss_reduce true --device cuda --trainer GradientTraceTrainer

#U373
python3 main.py --experiment U373 --dataset U373 --data_dir ./data/U373 --weight_loss false --weight_type none --arch UNet --trainer GradientTraceTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30 --device cuda --save_check_point false  --force_save_last true --deep_supervision false --optimizer SGD --learning-rate 1e-2 --loss WeightCrossEntropyLoss --loss_reduce true --device_id 0

python3 main.py --experiment U373_iw --dataset U373 --data_dir ./data/U373 --weight_loss true --weight_type single_baseline_weight --arch UNet --trainer GradientTraceTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30 --device cuda --save_check_point false  --force_save_last true --deep_supervision false --optimizer SGD --learning-rate 1e-2 --loss WeightCrossEntropyLoss --loss_reduce true --device_id 0

#HeLa
python3 main.py --experiment HeLa --dataset HeLa --data_dir ./data/HeLa --weight_loss false --weight_type none --arch UNet --trainer GradientTraceTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30 --device cuda --save_check_point false  --force_save_last true --deep_supervision false --optimizer SGD --learning-rate 1e-2 --loss WeightCrossEntropyLoss --loss_reduce true --device_id 0

python3 main.py --experiment HeLa_iw --dataset HeLa --data_dir ./data/HeLa --weight_loss true --weight_type single_baseline_weight --arch UNet --trainer GradientTraceTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30 --device cuda --save_check_point false  --force_save_last true --deep_supervision false --optimizer SGD --learning-rate 1e-2 --loss WeightCrossEntropyLoss --loss_reduce true --device_id 0
