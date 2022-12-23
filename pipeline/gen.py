import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer
import argparse
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from rouge import Rouge
import logging
import pdb


logger = logging.getLogger(__name__)

# BLEU
def _bleu_score(reference, hypothesis, smooth=None):
    # Pre-process
    references = [word_tokenize(reference)]
    hypothesis = word_tokenize(hypothesis)

    # (BLEU-1 + BLUE-2 + BLUE-3 + BLUE-4) / 4
    score = sentence_bleu(references, hypothesis, smoothing_function=smooth.method1)

    return score


# METEOR
def _meteor_score(reference, hypothesis):
    # Pre-process
    references = [word_tokenize(reference)]
    hypothesis = word_tokenize(hypothesis)

    score = meteor_score(references, hypothesis)

    return score


# ROUGE
def _rouge_score(reference, hypothesis, rouger=None):
    # ROUGE-2 F
    # rouger = Rouge()
    try:
        scores = rouger.get_scores(hypothesis, reference)
        score = scores[0]["rouge-2"]["f"]
    except ValueError as e:
        score = 0.0
    return score


def generate(args):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

    # tokenizer and model
    if args.model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif args.model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(device)

    # generate simple questions
    def get_decomposition(question):
        if args.model_type == "t5":
            input_text = "Paraphrase: " + question
        else:
            input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(device)
        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=args.max_tgt_len)
        # TODO: fix id=2537 ("�") bug
        # res = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        # enc = tokenizer.encode(res)
        # if 2537 in enc:
        #     enc.remove(2537)
        # res = tokenizer.decode(enc, skip_special_tokens=True).strip()
        
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    with open(args.src_path) as f:
        dataset = json.load(f)
    
    # Pre-process metrics
    smooth = SmoothingFunction()
    rouger = Rouge()

    # calculate metrics
    bleu, meteor, rouge = 0.0, 0.0, 0.0
    for i, data in enumerate(dataset):
        question = data["question"]
        label = ""
        for j, decomp in enumerate(data["question_decomposition"]):
            if j == 0:
                label += decomp["question"]
            else:
                label += " ; " + decomp["question"]
        prediction = get_decomposition(question)
        # Plain Text Process
        # prediction = prediction.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        data["decomposition"] = prediction
        _bleu = _bleu_score(label, prediction, smooth=smooth)
        _meteor = _meteor_score(label, prediction)
        _rouge = _rouge_score(label, prediction, rouger=rouger)
        bleu += _bleu
        meteor += _meteor
        rouge += _rouge
        # Print
        print("***** Case No.%d *****" % (i + 1))
        print("BLEU Score: %f" % _bleu)
        print("METEOR Score: %f" % _meteor)
        print("ROUGE Score: %f" % _rouge)

    bleu = bleu / len(dataset)
    meteor = meteor / len(dataset)
    rouge = rouge / len(dataset)

    with open(args.dest_path, "w") as f:
        json.dump(dataset, f, indent=4)

    return bleu, meteor, rouge


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--src_path", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--dest_path", default=None, type=str, required=True,
                        help="The input data dir.")

    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_src_len", default=70, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_tgt_len", default=100, type=int,
                        help="Optional target sequence length after tokenization.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    args = parser.parse_args()

    dest_dir = "/".join(args.dest_path.split("/")[:-1])
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    
    result = generate(args)
    bleu, meteor, rouge = result

    # logger.info("***** Final Result *****")
    # logger.info("Final BLEU: %f", bleu)
    # logger.info("Final METEOR: %f", meteor)
    # logger.info("Final ROUGE: %f", rouge)

    print("***** Final Result *****")
    print("Final BLEU: %f" % bleu)
    print("Final METEOR: %f" % meteor)
    print("Final ROUGE: %f" % rouge)


if __name__ == "__main__":
    main()
