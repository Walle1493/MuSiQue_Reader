export QUESTION_SINGLE=facebook/dpr-question_encoder-single-nq-base
export CONTEXT_SINGLE=facebook/dpr-ctx_encoder-single-nq-base
export QUESTION_MULTISET=facebook/dpr-question_encoder-multiset-base
export CONTEXT_MULTISET=facebook/dpr-ctx_encoder-multiset-base
export FILE_PATH=/home/mxdong/Data/MuSiQue/short_data/dev.json


CUDA_VISIBLE_DEVICES=0 python run_dpr.py \
    --question_model ${QUESTION_MULTISET} \
    --context_model ${CONTEXT_MULTISET} \
    --file_path ${FILE_PATH} \
