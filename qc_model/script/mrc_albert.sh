# export DATA_DIR=/home/mxdong/Data/MuSiQue/short_data
export DATA_DIR=/home/mxdong/Data/MuSiQue/format_data

export CLASSIFIER_NAME=/home/mxdong/Model/Classification/Binary/microsoft/deberta-v3-large

export SEQUENCE_GENERATOR_NAME=/home/mxdong/Model/Decomposition/MuSiQue/Sequence
export PARALLEL_GENERATOR_NAME=/home/mxdong/Model/Decomposition/MuSiQue/Parallel
export GENERATOR_TOKENIZER=facebook/bart-large

export RETRIEVER_NAME=/home/mxdong/Model/Selector/MuSiQue_Title/albert-xxlarge-v2
export READER_NAME=/home/mxdong/Model/Reader/MuSiQue/albert-xxlarge-v2

export RENAME=COMB04
export OUTPUT_DIR=/home/mxdong/Model/Four/${RENAME}


# Albert-xxLarge and Albert-xxLarge
CUDA_VISIBLE_DEVICES=3 python ../mrc.py \
    --classifier_type deberta \
    --classifier_name_or_path ${CLASSIFIER_NAME} \
    --generator_type bart \
    --seqgen_name_or_path ${SEQUENCE_GENERATOR_NAME} \
    --paragen_name_or_path ${PARALLEL_GENERATOR_NAME} \
    --generator_tokenizer ${GENERATOR_TOKENIZER} \
    --retriever_type albert \
    --retriever_name_or_path ${RETRIEVER_NAME} \
    --reader_type albert \
    --reader_name_or_path ${READER_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --file_name dev.json \
    --output_dir ${OUTPUT_DIR} \
    --max_cls_length 160 \
    --max_gen_length 48 \
    --max_seq_length 512 \
    --max_query_length 32 \
    --max_answer_length 30 \
    --n_best_size 20 \