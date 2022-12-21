# export DATA_DIR=/home/mxdong/Data/MuSiQue/pipeline_data/gen_data_short
export DATA_DIR=/home/mxdong/Data/MuSiQue/pipeline_data/gen_data

export RETRIEVER_NAME=/home/mxdong/Model/Selector/MuSiQue_Title/microsoft/deberta-v3-large
export READER_NAME=/home/mxdong/Model/Reader/MuSiQue/microsoft/deberta-v3-large
export RENAME=DEBERTA_AND_DEBERTA
export OUTPUT_DIR=/home/mxdong/Model/Pipeline/${RENAME}


# DebertaV3-Large and DebertaV3-Large
CUDA_VISIBLE_DEVICES=0 python ../mrc.py \
    --retriever_type deberta \
    --retriever_name_or_path ${RETRIEVER_NAME} \
    --reader_type deberta \
    --reader_name_or_path ${READER_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --file_name dev.json \
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length 512 \
    --max_query_length 32 \
    --max_answer_length 30 \
    --n_best_size 20 \

