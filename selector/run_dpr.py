import transformers
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import json
import argparse


def select(file_path, qTokenizer, qModel, cTokenizer, cModel, device):
    """Select the most relevant context with the question, calculate metrics meanwhile"""
    acc = 0.0
    count = 0
    with open(file_path) as f:
        dataset = json.load(f)
    for data in dataset:
        paragraphs = data["paragraphs"]
        context = [pargraph["title"] + ". " + pargraph["paragraph_text"] for pargraph in paragraphs]
        cFeature = cTokenizer(context, return_tensors="pt", padding=True).to(device)
        # obtain context output
        cOutput = cModel(
            input_ids=cFeature["input_ids"],
            attention_mask=cFeature["attention_mask"],
            token_type_ids=cFeature["token_type_ids"],
        ).pooler_output

        decompositions = data["question_decomposition"]
        for decomposition in decompositions:
            question = decomposition["question"]
            qFeature = qTokenizer(question, return_tensors="pt").to(device)
            # obtain question output
            qOutput = qModel(
                input_ids=qFeature["input_ids"],
                attention_mask=qFeature["attention_mask"],
                token_type_ids=qFeature["token_type_ids"],
            ).pooler_output
            # calculate similarity
            similarity = torch.cosine_similarity(qOutput, cOutput)
            pred = torch.argmax(similarity)
            gold = decomposition["paragraph_support_idx"]
            count += 1
            if pred == gold:
                acc += 1
    return acc / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_model", default=None, type=str, required=True)
    parser.add_argument("--context_model", default=None, type=str, required=True)
    parser.add_argument("--file_path", default=None, type=str, required=True)

    args = parser.parse_args()
    device = torch.device("cuda")

    question_model = DPRQuestionEncoder.from_pretrained(args.question_model).to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.question_model)
    context_model = DPRContextEncoder.from_pretrained(args.context_model).to(device)
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.context_model)

    acc = select(
        args.file_path,
        question_tokenizer,
        question_model,
        context_tokenizer,
        context_model,
        device,
    )

    print(acc)


if __name__ == "__main__":
    main()
