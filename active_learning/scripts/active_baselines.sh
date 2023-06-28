set -e

### change these variables if needed
DATA_DIR=data
TASK_NAME=dbpedia
MODEL_TYPE=bert
MODEL_NAME=/nlp/users/yekyung.kim/git/LINDA/LINDA/pretrained_models/bert-base-uncased #bert-base-uncased
COLDSTART=none
SAMPLING=bertKM

SEEDS='123 124 125 126 127' #'123 124 125 126 127'
if [ ${TASK_NAME} == 'rte' ] || [ ${TASK_NAME} == 'cola' ]
then
  EPOCH=10
else
  EPOCH=5
fi

if [ ${TASK_NAME} == 'rte' ]
then
  INCREMENT=100
  MAX_SIZE=1001
else
  INCREMENT=500
  MAX_SIZE=5001
fi

### end
for SEED in $SEEDS
do
  echo -e "\n\nSTARTING SEED $SEED \n\n"
  METHOD=${COLDSTART}-${SAMPLING}
  MODEL_DIR=models/${SEED}/${TASK_NAME}
  if [ "$COLDSTART" == "none" ]
  then
      MODEL0=$MODEL_NAME
      START=0
      METHOD=${SAMPLING}
  else
      MODEL0=${MODEL_DIR}/${COLDSTART}_${INCREMENT}
      START=$INCREMENT
  fi

  active (){
  # 1=number of samples
  # 2=model path
  # 3=sampling method
  echo -e "\n\nACQUIRING $1 SAMPLES WITH ${SAMPLING}\n\n"
  python -m src.active \
      --model_type $MODEL_TYPE \
      --model_name_or_path $2 \
      --task_name $TASK_NAME \
      --data_dir $DATA_DIR/$TASK_NAME \
      --output_dir ${MODEL_DIR}/${3}_${1} \
      --seed $SEED \
      --query_size $INCREMENT \
      --sampling $SAMPLING \
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
      --num_train_epochs $EPOCH \
      --output_dir $2 \
      --seed $SEED \
      --base_model $MODEL_NAME \
      --per_gpu_eval_batch_size 32 \
      --per_gpu_train_batch_size 32 \
      --evaluate_during_training
  }

  f=$MODEL0
  p=$(( $START + $INCREMENT ))
  while [ $p -le $MAX_SIZE ]
  do
    active $p $f $METHOD
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
done
