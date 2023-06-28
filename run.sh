DATASET=cola
SEED="1234 2345"
GPU=1
EPOCHS=3
BATCH_SIZE=16
BACKBONE=roberta_large # roberta_mc_large for winogrande, roberta_large for the others 

# Training of classifier 
#for seed in $SEED
#do
#  CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type xxxx_base --save_ckpt --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
#  CUDA_VISIBLE_DEVICES=$GPU python train.py --grad_accumulation 4 --train_type xxxx_base --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
#done

CUDA_VISIBLE_DEVICES=$GPU python construct_infoverse.py --train_type 1115_base_init --seed_list "1234 2345" --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed 1234 --backbone $BACKBONE
