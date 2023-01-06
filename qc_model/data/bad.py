import sys
import os
import json
import numpy as np
import re
import string
from collections import Counter
import pdb


def normalize_answer(s):
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def ans_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        f1 = 0.0
    else:
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
    em = (normalize_answer(prediction) == normalize_answer(ground_truth))
    return f1, em


def sup_score(prediction, gold):
    tp, fp, fn = 0, 0, 0
    for e in prediction:
        if e in gold:
            tp += 1
        else:
            fp += 1
    for e in gold:
        if e not in prediction:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return f1, em


def bad_analysis(data_path, sup_path, ans_path):
    """
    sup_f1 <= 0.5
    or ans f1 <= 0.5
    """

    with open(data_path) as f:
        dataset = json.load(f)
    sups = np.load(sup_path, allow_pickle=True)
    anss = np.load(ans_path, allow_pickle=True)
    
    # # 统计评分较低的案例
    # for data, sup, ans in zip(dataset, sups, anss):
    #     gold_sup = [decomp["paragraph_support_idx"] for decomp in data["question_decomposition"]]
    #     gold_ans = data["answer"]
    #     sup_f1, _ = sup_score(sup, gold_sup)
    #     ans_f1, _ = ans_score(ans, gold_ans)
    #     if sup_f1 <= 0.5:
    #         print("********************")
    #         print("id:", data["id"])
    #         print("sup_f1:", sup_f1)
    #         print("ans_f1:", ans_f1)
    #         print("pred_sup", sup)
    #         print("gold_sup", gold_sup)
    #         print("pred_ans", ans)
    #         print("gold_ans", gold_ans)
    
    # 统计不擅长哪类问题
    score_dict = {
        "2hop": [0.0, 0.0, 0.0, 0.0, 0],
        "3hop1": [0.0, 0.0, 0.0, 0.0, 0],
        "3hop2": [0.0, 0.0, 0.0, 0.0, 0],
        "4hop1": [0.0, 0.0, 0.0, 0.0, 0],
        "4hop2": [0.0, 0.0, 0.0, 0.0, 0],
        "4hop3": [0.0, 0.0, 0.0, 0.0, 0],
    }
    for data, sup, ans in zip(dataset, sups, anss):
        key = data["id"].split("_")[0]
        gold_sup = [decomp["paragraph_support_idx"] for decomp in data["question_decomposition"]]
        gold_ans = data["answer"]
        sup_f1, sup_em = sup_score(sup, gold_sup)
        ans_f1, ans_em = ans_score(ans, gold_ans)
        score_dict[key][0] += sup_f1
        score_dict[key][1] += sup_em
        score_dict[key][2] += ans_f1
        score_dict[key][3] += ans_em
        score_dict[key][4] += 1
    for key in score_dict:
        score_dict[key][0] /= score_dict[key][-1]
        score_dict[key][1] /= score_dict[key][-1]
        score_dict[key][2] /= score_dict[key][-1]
        score_dict[key][3] /= score_dict[key][-1]
    print(score_dict)

    return 


if __name__ == "__main__":

    # mode = sys.argv[1]  # train or dev

    data_path = "/home/mxdong/Data/MuSiQue/format_data/dev.json"
    sup_path = "/home/mxdong/Model/MultiStep/BART_AND_DEBERTA_AND_DEBERTA_2/supporting_prediction.npy"
    ans_path = "/home/mxdong/Model/MultiStep/BART_AND_DEBERTA_AND_DEBERTA_2/answer_prediction.npy"
    
    bad_analysis(data_path, sup_path, ans_path)
