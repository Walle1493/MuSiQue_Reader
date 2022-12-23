# export SRC_PATH=/home/mxdong/Data/MuSiQue/short_data/dev.json
export SRC_PATH=/home/mxdong/Data/MuSiQue/format_data/dev.json
export DEST_PATH=/home/mxdong/Data/MuSiQue/pipeline_data/gen_data/dev.json
# export MODEL_PATH=/home/mxdong/Model/Decomposition/MuSiQue/Bart-Large
export MODEL_PATH=/home/mxdong/Model/Seq2seq/MuSiQue/Bart-Large
export TOKENIZER_PATH=facebook/bart-large

# # seq2seq using ;
# CUDA_VISIBLE_DEVICES=0 python ../gen.py \
#     --model_type=bart \
#     --tokenizer_name=${TOKENIZER_PATH} \
#     --model_name_or_path=${MODEL_PATH} \
#     --src_path=${SRC_PATH} \
#     --dest_path=${DEST_PATH} \
#     --max_src_len 70\
#     --max_tgt_len 100\
#     --seed 42 \

# seq2seq using </s>
CUDA_VISIBLE_DEVICES=1 python ../gen.py \
    --model_type=bart \
    --tokenizer_name=${TOKENIZER_PATH} \
    --model_name_or_path=${MODEL_PATH} \
    --src_path=${SRC_PATH} \
    --dest_path=${DEST_PATH} \
    --max_src_len 70\
    --max_tgt_len 100\
    --seed 42 \