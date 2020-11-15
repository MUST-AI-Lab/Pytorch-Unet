#logit add----------------------------------------------------------------------------------------------------
#1
python3 train.py  --experiment logitadd_b1_m070_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b1_m080_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b1_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b1_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b1_m092_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b1_m093_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

#2
python3 train.py  --experiment logitadd_b2_m070_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distrubution --device_id 5

python3 train.py  --experiment logitadd_b2_m080_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b2_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b2_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b2_m092_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b2_m093_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

#4
python3 train.py  --experiment logitadd_b4_m070_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b4_m080_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b4_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b4_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b4_m092_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitadd_b4_m093_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitAddCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

#logit div----------------------------------------------------------------------------------------------------------------------------------------------------------
#1
python3 train.py  --experiment logitdiv_b1_m070_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b1_m080_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b1_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b1_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b1_m092_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b1_m093_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

#2
python3 train.py  --experiment logitdiv_b2_m070_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b2_m080_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b2_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b2_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b2_m092_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b2_m093_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 2 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

#4
python3 train.py  --experiment logitdiv_b4_m070_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.7 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b4_m080_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.8 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b4_m090_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b4_m091_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.91 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b4_m092_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.92 --save_check_point false  --force_save_last true  --weight_loss true --weight_type global_distrubution --device_id 5

python3 train.py  --experiment logitdiv_b4_m093_global --dataset Cam2007Dataset --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 4 --arch UNet --deep_supervision false --optimizer SGD   --loss LogitDivCELoss  --epochs 30 --learning-rate 1e-2  --momentum 0.93 --save_check_point false  --force_save_last true  --weight_loss true --weight_type single_distrubution --device_id 5

