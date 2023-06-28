SEED="1234"
GPU=1
EPOCHS=10
RATIO=1.0
BATCH_SIZE=8
BACKBONE=roberta_large
DATA=imp # sst5

for seed in $SEED
do
  CUDA_VISIBLE_DEVICES=$GPU python ./data_annotation/train_anno.py --annotation infoverse --save_ckpt --model_lr 1e-5 --grad_accumulation 2 --train_type 0612_base --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATA --seed $seed --backbone $BACKBONE  
done
