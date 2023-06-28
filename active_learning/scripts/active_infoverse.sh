set -e

### change these variables if needed
DATA_DIR=data
TASK_NAME=rte
MODEL_TYPE=bert
MODEL_NAME=bert-base-uncased
SEED=134
COLDSTART=none

INIT_SAMPLING=rand
SAMPLING=infoverse
DPP_SAMPLING=inv

ENS_SEED='1 2 3'

if [ ${TASK_NAME} == 'rte' ]
then
  EPOCH=10
  INCREMENT=100
  INIT_SIZE=101
  MAX_SIZE=1001
else
  EPOCH=5
  INCREMENT=500
  INIT_SIZE=501
  MAX_SIZE=5001
fi

METHOD=${COLDSTART}-${INIT_SAMPLING}
MODEL_DIR=models/${SEED}/${TASK_NAME}
if [ "$COLDSTART" == "none" ]
then
    MODEL0=$MODEL_NAME
    START=0
    METHOD=${INIT_SAMPLING}
else
    MODEL0=${MODEL_DIR}/${COLDSTART}_${INCREMENT}
    START=$INCREMENT
fi

active (){
# 1=number of samples
# 2=model path
# 3=sampling method
echo -e "\n\nACQUIRING $1 SAMPLES\n\n"
python -m src.active \
    --model_type $MODEL_TYPE \
    --model_name_or_path $2 \
    --task_name $TASK_NAME \
    --data_dir $DATA_DIR/$TASK_NAME \
    --output_dir ${MODEL_DIR}/${3}_${1} \
    --seed $SEED \
    --query_size $INCREMENT \
    --sampling $INIT_SAMPLING \
    --base_model $MODEL_NAME \
    --per_gpu_eval_batch_size 32 \
    --max_seq_length 128
}

train (){
# 1 = number of samples
# 2 = output directory
echo -e "\n\nTRAINING WITH $1 SAMPLES\n\n"
python -m src.train \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_test \
    --data_dir $DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs ${EPOCH} \
    --output_dir $2 \
    --seed $SEED \
    --base_model $MODEL_NAME \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 32 
}

f=$MODEL0
p=$(( $START + $INCREMENT ))
while [ $p -le $INIT_SIZE ]
do
    active $p $f $METHOD
    f=${MODEL_DIR}/${METHOD}_$p
    train $p $f
    rm ${f}/pytorch_model.bin
    rm ${f}/vocab.txt
    p=$(( $p + $INCREMENT ))
done

set -e

COLDSTART=rand

METHOD=${COLDSTART}-${SAMPLING}_${DPP_SAMPLING}
MODEL_DIR=models/${SEED}/${TASK_NAME}
if [ "$COLDSTART" == "none" ]
then
    MODEL0=$MODEL_NAME
    START=0
    METHOD=${SAMPLING}_${DPP_SAMPLING}
else
    MODEL0=${MODEL_DIR}/${COLDSTART}_${INCREMENT}
    START=$INCREMENT
fi

## training multiple model and save model each epoch before starting sampling
# (currently, it re-write $MODEL0's saved model and have potential problems)
ens_train (){
# 1 = output directory
for e_seed in $ENS_SEED
do
  echo -e "\n\nTRAINING $e_seed seed for sampled data of $1 \n\n"
  python -m src.train \
      --model_type $MODEL_TYPE \
      --model_name_or_path $MODEL_NAME \
      --task_name $TASK_NAME \
      --do_train \
      --do_test \
      --data_dir $DATA_DIR/$TASK_NAME \
      --max_seq_length 128 \
      --learning_rate 2e-5 \
      --num_train_epochs ${EPOCH} \
      --output_dir $1 \
      --seed ${e_seed} \
      --base_model $MODEL_NAME \
      --per_gpu_eval_batch_size 32 \
      --per_gpu_train_batch_size 32 \
      --save_model_during_training
done
}
active (){
# 1=number of samples
# 2=model path
# 3=sampling method
echo -e "\n\nACQUIRING $1 SAMPLES\n\n"
python -m src_infoverse.sample_infoverse \
    --model_type $MODEL_TYPE \
    --model_name_or_path $2 \
    --task_name $TASK_NAME \
    --data_dir $DATA_DIR/$TASK_NAME \
    --output_dir ${MODEL_DIR}/${3}_${1} \
    --seed $SEED \
    --query_size $INCREMENT \
    --sampling $SAMPLING \
    --dpp_sampling $DPP_SAMPLING \
    --base_model $MODEL_TYPE \
    --per_gpu_eval_batch_size 32 \
    --max_seq_length 128 \
    --seed_list "${ENS_SEED}"
}

train (){
# 1 = number of samples
# 2 = output directory
echo -e "\n\nTRAINING WITH $1 SAMPLES\n\n"
python -m src.train \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_test \
    --data_dir $DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs ${EPOCH} \
    --output_dir $2 \
    --seed $SEED \
    --base_model $MODEL_NAME \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 32
}

f=$MODEL0
p=$(( $START + $INCREMENT ))
while [ $p -le $MAX_SIZE ]
do
    ens_train $f
    active $p $f $METHOD

    echo -e "\n\n Remove Ensemble Model\n\n"
    for e_seed in $ENS_SEED
    do
      rm -rf $f/seed${e_seed}_*
    done

    f=${MODEL_DIR}/${METHOD}_$p
    train $p $f
    p=$(( $p + $INCREMENT ))
done

echo -e "\nALL ROUND COMPLETED\n"
echo -e "\nREMOVING ALL SAVED CHECKPOINT \n"

f=$MODEL0
p=$(( $START + $INCREMENT ))
while [ $p -le $MAX_SIZE ]
do
  f=${MODEL_DIR}/${METHOD}_$p
  if [ ${p} -gt ${INCREMENT} ]; then
    echo -e "\n REMOVING ${f}/pytorch_model.bin \n"
    rm ${f}/pytorch_model.bin
    rm ${f}/vocab.txt
  fi

  p=$(( $p + $INCREMENT ))
done
