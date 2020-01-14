from flask import json
from tqdm import tqdm


def get_fever_dataset(get_train=True, get_dev=False, limit=-1):
    if get_train:
        file_path = "./data/train/train.jsonl"
    elif get_dev:
        file_path = "./data/shared_task_dev.jsonl"
    values = []
    print("Loading test cases")
    i = 0
    with open(file_path, "r") as fp:
        for line in tqdm(fp.readlines()):
            if i > limit > 0:
                break
            obj = json.loads(line)
            if obj["verifiable"] == 'VERIFIABLE':
                i += 1
                values.append((obj['claim'], obj['evidence']))
    return values


if __name__ == "__main__":
    pass