# export DATA_DIR=/home/mxdong/Data/MuSiQue/short_data
export DATA_DIR=/home/mxdong/Data/MuSiQue/format_data

export GENERATOR_NAME=/home/mxdong/Model/MultiStep/MuSiQue/Bart-Large-eos-32-2
export GENERATOR_TOKENIZER=facebook/bart-large
export RETRIEVER_NAME=/home/mxdong/Model/Selector/MuSiQue_Title/albert-xxlarge-v2
export READER_NAME=/home/mxdong/Model/Reader/MuSiQue/microsoft/deberta-v3-large
export RENAME=BART_AND_ALBERT_AND_DEBERTA_2
export OUTPUT_DIR=/home/mxdong/Model/MultiStep/${RENAME}


# Albert-xxLarge and DebertaV3-Large
CUDA_VISIBLE_DEVICES=3 python ../mrc.py \
    --generator_type bart \
    --generator_tokenizer ${GENERATOR_TOKENIZER} \
    --generator_name_or_path ${GENERATOR_NAME} \
    --retriever_type albert \
    --retriever_name_or_path ${RETRIEVER_NAME} \
    --reader_type deberta \
    --reader_name_or_path ${READER_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --file_name dev.json \
    --output_dir ${OUTPUT_DIR} \
    --max_gen_length 48 \
    --max_seq_length 512 \
    --max_query_length 32 \
    --max_answer_length 30 \
    --n_best_size 20 \

