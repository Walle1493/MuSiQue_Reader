import sys
import os
import json
import random


def create_multi_choice_dataset(src_path, dest_path, neg_num=3):
    new = []
    
    with open(src_path) as f:
        dataset = json.load(f)
    
    for data in dataset:
        _id = data["id"]
        paragraphs = data["paragraphs"]
        decompositions = data["question_decomposition"]

        for i, decomposition in enumerate(decompositions):
            curr = {}
            question = decomposition["question"]

            # positive index
            pos_idx = decomposition["paragraph_support_idx"]
            # negative indices
            neg_idxs = list(range(len(paragraphs)))
            neg_idxs.remove(pos_idx)
            random.shuffle(neg_idxs)
            neg_idxs = neg_idxs[:neg_num]

            curr["id"] = _id + "_" + str(i+1)
            
            # positive label
            curr["label"] = random.randint(0, neg_num)
            curr["context"] = []
            for i in range(neg_num + 1):
                if i == curr["label"]:
                    para = paragraphs[pos_idx]
                else:
                    para = paragraphs[neg_idxs.pop(0)]
                curr["context"].append(para["paragraph_text"])
                # curr["context"].append(para["title"] + ". " + para["paragraph_text"])

            # question: #x -> ANS
            if "#" in question:
                if "#1" in question:
                    question = question.replace("#1", decompositions[0]["answer"])
                if "#2" in question:
                    question = question.replace("#2", decompositions[1]["answer"])
                if "#3" in question:
                    question = question.replace("#3", decompositions[2]["answer"])

            curr["question"] = question

            new.append(curr)

    with open(dest_path, "w") as f:
        json.dump(new, f, indent=4)


if __name__ == "__main__":

    mode = sys.argv[1]  # train or dev
    # neg_num = sys.argv[2]   # 3

    src_path = "/home/mxdong/Data/MuSiQue/format_data"
    dest_path = "/home/mxdong/Data/MuSiQue/multi_choice_data"
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    src_path = os.path.join(src_path, mode + ".json")
    dest_path = os.path.join(dest_path, mode + ".json")
    
    create_multi_choice_dataset(src_path, dest_path)
