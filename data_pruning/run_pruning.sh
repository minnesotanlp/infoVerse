DATASET=cola
SEED=1234
GPU=1
EPOCHS=10
RATIO=0.09
BATCH_SIZE=16
BACKBONE=roberta_large

CUDA_VISIBLE_DEVICES=$GPU python ./data_pruning/train_pruning.py --train_type 1115_infoverse_dpp --save_ckpt --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $SEED --backbone $BACKBONE
