import sys
import os
import json
import pdb


def shorten(src_path, dest_path):
    """提取前100条数据"""

    with open(src_path) as f:
        dataset = json.load(f)
    
    with open(dest_path, "w") as f:
        json.dump(dataset[:100], f, indent=4)
    
    return 


if __name__ == "__main__":

    mode = sys.argv[1]  # train or dev

    src_path = "/home/mxdong/Data/MuSiQue/multi_choice_title"
    dest_path = "/home/mxdong/Data/MuSiQue/multi_choice_short"

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    src_path = os.path.join(src_path, mode + ".json")
    dest_path = os.path.join(dest_path, mode + ".json")
    
    shorten(src_path, dest_path)
