import sys
import os
import json
import pdb


def create_simgle_hop_dataset(src_path, dest_path):
    new = []
    
    with open(src_path) as f:
        dataset = json.load(f)
    
    for data in dataset:
        _id = data["id"]
        paragraphs = data["paragraphs"]
        # question = data["question"]
        decompositions = data["question_decomposition"]
        # answer = data["answer"]
        # answer_aliases = data["answer_aliases"]
        # answerable = data["answerable"]

        for i, decomposition in enumerate(decompositions):
            curr = {}
            question = decomposition["question"]
            answer = decomposition["answer"]
            sup_idx = decomposition["paragraph_support_idx"]

            curr["id"] = _id + "_" + str(i+1)
            curr["context"] = paragraphs[sup_idx]["paragraph_text"]

            # question: #x -> ANS
            if "How were the the Portuguese expelled from #2" in question:
                pdb.set_trace()
            if "#" in question:
                if "#1" in question:
                    question = question.replace("#1", decompositions[0]["answer"])
                if "#2" in question:
                    question = question.replace("#2", decompositions[1]["answer"])
                if "#3" in question:
                    question = question.replace("#3", decompositions[2]["answer"])

            curr["question"] = question
            curr["answer"] = answer

            # start
            curr["start"] = curr["context"].index(answer)

            new.append(curr)

    with open(dest_path, "w") as f:
        json.dump(new, f, indent=4)


if __name__ == "__main__":

    mode = "train"
    # mode = "dev"

    src_path = "/home/mxdong/Data/MuSiQue/format_data"
    dest_path = "/home/mxdong/Data/MuSiQue/single_hop_data"
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    src_path = os.path.join(src_path, mode + ".json")
    dest_path = os.path.join(dest_path, mode + ".json")
    
    create_simgle_hop_dataset(src_path, dest_path)
