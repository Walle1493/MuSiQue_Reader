export DATA_DIR=/home/mxdong/Data/MuSiQue/single_hop_data
export TASK_NAME=MuSiQue
export MODEL_NAME=bert-base-uncased
export OUTPUT_DIR=/home/mxdong/Model/Reader/${TASK_NAME}/${MODEL_NAME}

CUDA_VISIBLE_DEVICES=0 python run_musique.py \
    --model_type bert \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --train_file train.json \
    --predict_file dev.json \
    --max_seq_length 512 \
    --doc_stride 256 \
    --max_query_length 32 \
    --per_gpu_train_batch_size 32   \
    --per_gpu_eval_batch_size 32   \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --logging_steps 200 \
    --save_steps 2000 \
    --adam_epsilon 1e-8 \
    --warmup_steps 300 \
    --overwrite_output_dir \
    # --evaluate_during_training \