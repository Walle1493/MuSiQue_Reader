from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import glob
import json
import timeit
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import re
import string
from collections import Counter
import pdb

from transformers import (
    BertConfig, BertForMultipleChoice, BertForQuestionAnswering, BertTokenizer,
    XLMConfig, XLMForMultipleChoice, XLMForQuestionAnswering, XLMTokenizer, 
    XLNetConfig, XLNetForMultipleChoice, XLNetForQuestionAnswering, XLNetTokenizer,
    DistilBertConfig, DistilBertForMultipleChoice, DistilBertForQuestionAnswering, DistilBertTokenizer,
    AlbertConfig, AlbertForMultipleChoice, AlbertForQuestionAnswering, AlbertTokenizer,
    DebertaV2Config, DebertaV2ForMultipleChoice, DebertaV2ForQuestionAnswering, DebertaV2Tokenizer,
)

RETRIEVER_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForMultipleChoice, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMultipleChoice, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer),
    'deberta': (DebertaV2Config, DebertaV2ForMultipleChoice, DebertaV2Tokenizer),
}

READER_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    'deberta': (DebertaV2Config, DebertaV2ForQuestionAnswering, DebertaV2Tokenizer),
}

logger = logging.getLogger(__name__)


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


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


def evaluate(args, retriever, reader, retriever_tokenizer, reader_tokenizer):
    file_path = os.path.join(args.data_dir, args.file_name)
    with open(file_path) as f:
        dataset = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Eval!
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(dataset))
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(dataset)))

    pred_supports, pred_answers = [], []
    gold_supports, gold_answers = [], []
    start_time = timeit.default_timer()
    
    for i, data in enumerate(dataset):
        # gold support and gold answer
        gold_support = [decomp["paragraph_support_idx"] for decomp in data["question_decomposition"]]
        gold_answer = data["answer"]
        gold_supports.append(gold_support)
        gold_answers.append(gold_answer)

        # prediction for curr data
        sup, ans = [], []
        decompositions = data["decomposition"].split(";")
        for idx, query in enumerate(decompositions):
            # Retriever
            # modify #x in each decomp
            pos = -1
            pos = query.find("#", pos + 1)
            while pos != -1:
                num = int(query[pos + 1])
                if num >= idx + 1:
                    try:
                        query = query.replace(query[pos:pos+2], ans[-1])
                    except:
                        query = query.replace(query[pos:pos+2], "")
                else:
                    try:
                        query = query.replace(query[pos:pos+2], ans[num-1])
                    except:
                        query = query.replace(query[pos:pos+2], "")
                pos = query.find("#", pos + 1)
            # encode 20 (query, para) pairs
            features = []
            for paragraph in data["paragraphs"]:
                # text = paragraph["paragraph_text"]
                text = paragraph["title"] + ". " + paragraph["paragraph_text"]
                feature = retriever_tokenizer.encode_plus(
                    query, text, 
                    add_special_tokens=True, 
                    max_length=args.max_seq_length,
                )
                padding_length = args.max_seq_length - len(feature["input_ids"])
                # pad_token = 0
                feature["input_ids"] = feature["input_ids"] + ([0] * padding_length)
                feature["attention_mask"] = feature["attention_mask"] + ([0] * padding_length)

                assert len(feature["input_ids"]) == args.max_seq_length
                assert len(feature["attention_mask"]) == args.max_seq_length
                features.append(feature)

            input_ids = torch.tensor(
                [feature["input_ids"] for feature in features], 
                dtype=torch.long
            ).unsqueeze(0).to(args.device)
            attention_masks = torch.tensor(
                [feature["attention_mask"] for feature in features], 
                dtype=torch.long
            ).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                }
                # 测试multiple-choice的输入维度
                outputs = retriever(**inputs)
                logits = outputs.logits
            
            text_idx = logits.detach().cpu().numpy()
            text_idx = np.argmax(text_idx)
            sup.append(text_idx)
            text = data["paragraphs"][text_idx]["paragraph_text"]
            # doc = data["paragraphs"][doc_idx]["title"] + ". " + data["paragraphs"][doc_idx]["paragraph_text"]

            # Reader
            truncated_query = reader_tokenizer.encode(
                query, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=args.max_query_length
            )
            tokens = reader_tokenizer.tokenize(query + " " + text)
            feature = reader_tokenizer.encode_plus(
                query, text, 
                add_special_tokens=True, 
                max_length = args.max_seq_length,
            )
            padding_length = args.max_seq_length - len(feature["input_ids"])
            feature["input_ids"] = feature["input_ids"] + ([0] * padding_length)
            feature["attention_mask"] = feature["attention_mask"] + ([0] * padding_length)

            assert len(feature["input_ids"]) == args.max_seq_length
            assert len(feature["attention_mask"]) == args.max_seq_length

            input_ids = torch.tensor(
                feature["input_ids"], 
                dtype=torch.long
            ).unsqueeze(0).to(args.device)
            attention_masks = torch.tensor(
                feature["attention_mask"], 
                dtype=torch.long
            ).unsqueeze(0).to(args.device)

            with torch.no_grad():
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                }
                outputs = reader(**inputs)
                start_logits = outputs.start_logits.detach().cpu().tolist()[0]
                end_logits = outputs.end_logits.detach().cpu().tolist()[0]

            start_indexes = _get_best_indexes(start_logits, args.n_best_size)
            end_indexes = _get_best_indexes(end_logits, args.n_best_size)
            results = []
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(tokens) + 3:
                        continue
                    if end_index >= len(tokens) + 3:
                        continue
                    if start_index < len(truncated_query) + 2:
                        continue
                    if end_index < len(truncated_query) + 2:
                        continue
                    # TODO: <token_to_orig_map>
                    # if args.reader_type in ["deberta", "albert"]:
                    #     if reader_tokenizer.convert_ids_to_tokens(feature["input_ids"][start_index])[0].isalpha() and \
                    #         reader_tokenizer.convert_ids_to_tokens(feature["input_ids"][start_index-1])[0].isalpha():
                    #         continue
                    #     if reader_tokenizer.convert_ids_to_tokens(feature["input_ids"][end_index])[0].isalpha() and \
                    #         reader_tokenizer.convert_ids_to_tokens(feature["input_ids"][end_index-1])[0].isalpha():
                    #         continue
                    # if not feature.token_is_max_context.get(start_index, False):
                    #     continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_answer_length:
                        continue
                    results.append(
                        (
                            start_index,
                            end_index,
                            start_logits[start_index],
                            end_logits[end_index],
                        )
                    )
            results = sorted(results, key=lambda x: (x[2] + x[3]), reverse=True)
            best_start_index, best_end_index = results[0][:2]
            pred = reader_tokenizer.convert_ids_to_tokens(feature["input_ids"][best_start_index:best_end_index+1])
            pred = reader_tokenizer.convert_tokens_to_string(pred)
            ans.append(pred)

        print("Prediction for Question {}:".format(i + 1))
        print("Supporting: {}".format(sup))
        print("Answer: {}".format(ans))
        pred_supports.append(sup)
        pred_answers.append(ans[-1])

    # Metrics
    metrics = {
        "sup_f1": 0.0,
        "sup_em": 0.0,
        "ans_f1": 0.0,
        "ans_em": 0.0,
    }
    for (pred_sup, pred_ans, gold_sup, gold_ans) in zip(
        pred_supports, pred_answers, gold_supports, gold_answers
    ):
        sup_f1, sup_em = sup_score(pred_sup, gold_sup)
        ans_f1, ans_em = ans_score(pred_ans, gold_ans)
        metrics["sup_f1"] += sup_f1
        metrics["sup_em"] += sup_em
        metrics["ans_f1"] += ans_f1
        metrics["ans_em"] += ans_em
    metrics["sup_f1"] /= len(dataset)
    metrics["sup_em"] /= len(dataset)
    metrics["ans_f1"] /= len(dataset)
    metrics["ans_em"] /= len(dataset)

    evalTime = timeit.default_timer() - start_time
    # logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    print("  Evaluation done in total {} secs ({} sec per example)".format(evalTime, evalTime / len(dataset)))
    # Compute predictions
    supporting_prediction_file = os.path.join(args.output_dir, "supporting_prediction.npy")
    answer_prediction_file = os.path.join(args.output_dir, "answer_prediction.npy")
    np.save(supporting_prediction_file, pred_supports)
    np.save(answer_prediction_file, pred_answers)

    # logger.info("***** Eval result *****")
    print("***** Eval result *****")
    for key in metrics.keys():
        # logger.info("  %s = %s", key, str(metrics[key]))
        print("  {} = {}".format(key, str(metrics[key])))
        
    return metrics


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--retriever_type", default=None, type=str, required=True)
    parser.add_argument("--retriever_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--reader_type", default=None, type=str, required=True)
    parser.add_argument("--reader_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--file_name", default="", type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_query_length", default=32, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)

    args = parser.parse_args()

    # Setup cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Load Retriever
    args.retriever_type = args.retriever_type.lower()
    retriever_config_class, retriever_model_class, retriever_tokenizer_class = RETRIEVER_CLASSES[args.retriever_type]
    retriever_config = retriever_config_class.from_pretrained(
        args.retriever_name_or_path,
        cache_dir=None
    )
    retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
        args.retriever_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None
    )
    retriever = retriever_model_class.from_pretrained(
        args.retriever_name_or_path,
        from_tf=bool('.ckpt' in args.retriever_name_or_path),
        config=retriever_config,
        cache_dir=None
    )
    retriever.to(args.device)

    # Load Reader
    args.reader_type = args.reader_type.lower()
    reader_config_class, reader_model_class, reader_tokenizer_class = READER_CLASSES[args.reader_type]
    reader_config = reader_config_class.from_pretrained(
        args.reader_name_or_path,
        cache_dir=None
    )
    reader_tokenizer = reader_tokenizer_class.from_pretrained(
        args.reader_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None
    )
    reader = reader_model_class.from_pretrained(
        args.reader_name_or_path,
        from_tf=bool('.ckpt' in args.reader_name_or_path),
        config=reader_config,
        cache_dir=None
    )
    reader.to(args.device)


    # Evaluate
    results = evaluate(
        args, 
        retriever, reader, 
        retriever_tokenizer, reader_tokenizer, 
    )
    return results


if __name__ == "__main__":
    main()
