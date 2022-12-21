import sys
import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import pdb


def j_greaterThan_i(path):
    """
    q[i]: xxx #j xxx
    其中j大于等于i（从1开始计数）
    """

    with open(path) as f:
        dataset = json.load(f)
    
    gt_count = 0
    detail_counts = [0, 0, 0, 0]   # #1, #2, #3 in q[1], q[2], q[3]
    for data in dataset:
        decomposition = data["decomposition"]
        _id = data["id"]
        for i, decomp in enumerate(decomposition.split(";")):
            if "#" in decomp:
                pos = decomp.index("#") + 1
                j = int(decomp[pos])
                if j >= i + 1:
                    # gt_count += 1
                    detail_counts[i] += 1
                    # break
    
    print(gt_count) # 26
    print(detail_counts)
    
    return 


if __name__ == "__main__":

    # mode = sys.argv[1]  # train or dev

    path = "/home/mxdong/Data/MuSiQue/pipeline_data/gen_data"
    path = os.path.join(path, "dev.json")
    
    j_greaterThan_i(path)
