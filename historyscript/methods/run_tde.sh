#TDE
#ce
python3 train_tde.py  --experiment ce_tdev5 --dataset Cam2007DatasetV2 --data_dir  ./data/Cam2007_n  --input_channels 3  --num_classes 32 --batch-size 1 --arch UNetTDEv5 --deep_supervision false --optimizer SGD  --loss WeightCrossEntropyLoss  --epochs 30 --learning-rate 1e-2  --momentum 0.9 --save_check_point false  --force_save_last true  --weight_loss false --weight_type none --device_id 0 --loss_reduce true


