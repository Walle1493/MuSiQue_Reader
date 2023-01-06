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


def select(src_path, dest_path):
    """各种类型数据各取2条，共12条"""

    with open(src_path) as f:
        dataset = json.load(f)
    
    new_dataset= []
    type_dict = {
        "2hop": 0,
        "3hop1": 0,
        "3hop2": 0,
        "4hop1": 0,
        "4hop2": 0,
        "4hop3": 0,
    }

    for data in dataset:
        _type = data["id"].split("_")[0]
        type_dict[_type] += 1
        if type_dict[_type] <= 2:
            new_dataset.append(data)
    
    with open(dest_path, "w") as f:
        json.dump(new_dataset, f, indent=4)
    
    return 


if __name__ == "__main__":

    mode = sys.argv[1]  # train or dev

    src_path = "/home/mxdong/Data/MuSiQue/format_data"
    dest_path = "/home/mxdong/Data/MuSiQue/short_data"

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    src_path = os.path.join(src_path, mode + ".json")
    dest_path = os.path.join(dest_path, mode + ".json")
    
    select(src_path, dest_path)
