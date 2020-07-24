set -ex
CUDA_VISIBLE_DEVICES=1 python extract_workshop_baseline_2d.py --dataroot dataset/test/ --name workshop_baseline_notexture_tuning_v1 --model retrieval_workshop_baseline_tuning --dataset_mode retrieval_workshop_baseline_eval_2d --crop_size 256 --fine_size 256 #--epoch 20
# --epoch 30
