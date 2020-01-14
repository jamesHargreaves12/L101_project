import json
import os
from nltk import tokenize
import jsonlines as jsonlines
from tqdm import tqdm

outloc = "data/first_sentences/train_2.jsonl"
with jsonlines.open(outloc, mode='w') as writer:
    parent = "data/wiki-pages/wiki-pages/"
    for child in tqdm(os.listdir(parent)):
        fp = open(os.path.join(parent,child), "r")
        for line in fp.readlines():
            val = json.loads(line)
            new_val = {}
            new_val["id"] = val["id"]
            sentences = tokenize.sent_tokenize(val["text"])
            if len(sentences) > 0:
                new_val["text"] = sentences[0]
            else:
                # print(val)
                # print(sentences)
                new_val["text"] = ""
            writer.write(new_val)

