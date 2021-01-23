#HeLa
python3 main.py --experiment HeLa_weight --dataset HeLa --data_dir ./data/HeLa --weight_loss true --weight_type single_baseline_weight --arch UNet --trainer STDTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30

python3 main.py --experiment HeLa_weight --dataset HeLa --data_dir ./data/HeLa --weight_loss true --weight_type single_count --arch UNet --trainer CBLossTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30

python3 main.py --experiment HeLa --dataset HeLa --data_dir ./data/HeLa --weight_loss false --weight_type none --arch UNet --trainer STDTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30

#U373
python3 main.py --experiment U373_weight --dataset HeLa --data_dir ./data/U373 --weight_loss true --weight_type single_baseline_weight --arch UNet --trainer STDTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30

python3 main.py --experiment U373_weight --dataset HeLa --data_dir ./data/U373 --weight_loss true --weight_type single_count --arch UNet --trainer CBLossTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30

python3 main.py --experiment U373 --dataset HeLa --data_dir ./data/U373 --weight_loss false --weight_type none --arch UNet --trainer STDTrainer --input_channels 1 --num_classes 2 --momentum 0.99 --device cuda --epochs 30
