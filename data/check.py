import sys
import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import pdb


def check_no_sharp(path):
    """检查question中是否还存在#符号"""

    with open(path) as f:
        dataset = json.load(f)
    
    sharp_count = 0
    for data in dataset:
        question = data["question"]
        if "#" in question:
            sharp_count += 1
            print(question)
    
    print(sharp_count)
    # train = 2; #9 Dream
    # dev = 0
    
    return 


def count_data_total(path):
    """统计演化成单跳问题后，各跳问题有多少"""

    with open(path) as f:
        dataset = json.load(f)

    _2hop, _3hop, _4hop, total = 0, 0, 0, 0
    for data in dataset:
        _id = data["id"]
        if "2hop" in _id:
            _2hop += 1
        if "3hop" in _id:
            _3hop += 1
        if "4hop" in _id:
            _4hop += 1
        total += 1
    assert _2hop + _3hop + _4hop == total

    print("2-hop:", _2hop)
    print("3-hop:", _3hop)
    print("4-hop:", _4hop)
    print("Total:", total)

    return 


def count_doc_length(path):
    """统计文档长度"""

    with open(path) as f:
        dataset = json.load(f)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    doc_len = []
    for data in dataset:
        doc = data["context"]
        tokens = tokenizer.tokenize(doc)
        doc_len.append(len(tokens))
    
    np.save("doc_len", doc_len)
    print(doc_len)
    
    return 


def count_query_length(path):
    """统计问题长度"""

    with open(path) as f:
        dataset = json.load(f)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    query_len = []
    for data in dataset:
        query = data["question"]
        tokens = tokenizer.tokenize(query)
        query_len.append(len(tokens))
    
    np.save("query_len", query_len)
    print(query_len)
    
    return 


if __name__ == "__main__":

    mode = sys.argv[1]  # train or dev

    path = "/home/mxdong/Data/MuSiQue/single_hop_data"
    path = os.path.join(path, mode + ".json")
    
    count_query_length(path)
