from tqdm import tqdm
import os

from baseline.database import convert_to_form_in_db

os.environ["CLASSPATH"] = "./DrQA/data/corenlp/*"

from baseline.get_dataset import get_fever_dataset
from baseline.mediawiki_api import get_pageview_data
from baseline.results import get_possible_docs
from baseline.utils import get_overlap, extract_titles_from_evidence, get_mlp_features
from running_models.run_DrQA import get_drqa_score_sent, get_drqa_score_doc


def get_unique_docs(docs):
    title_to_doc = {}
    for x in docs:
        title_to_doc[x["title"]] = x
    return list(title_to_doc.values())


# open("data/nn_train_with_drqa.csv", "w"),open('data/tmp_train_recorder_with_drqa.txt', "w")


data_file = open("data/nn_train_with_drqa_full_with_capital_ents_2.csv", "a+")
current_position_file = open('data/cur_pos_full_with_ents_2.txt', "r+")
current_position_file.seek(0)
current_doc = int(current_position_file.read() or 0)

train = []
size_dataset = -1
IS_TRAIN_DATASET = True
with_cap_ents = True
dataset = get_fever_dataset(get_train=IS_TRAIN_DATASET, get_dev=not IS_TRAIN_DATASET, limit=size_dataset)
print(data_file)
print(current_position_file)
print(IS_TRAIN_DATASET)
print(with_cap_ents)
print("Creating vectors")
print("Starting:", current_doc)
for claim, evidence in tqdm(dataset[current_doc:size_dataset]):
    ev_titles = extract_titles_from_evidence(evidence)
    possible_docs = get_possible_docs(claim, with_cap_ents)
    all_titles = [x["title"] for x in possible_docs]
    for doc in get_unique_docs(possible_docs):
        input_tup = get_mlp_features(doc, claim, all_titles)
        doc_title = convert_to_form_in_db(doc["title"])
        correct = 1 if doc_title in ev_titles else 0
        output = correct
        data_file.write(",".join(map(str, input_tup)) + "," + str(output) + "\n")
        data_file.flush()
        train.append((input_tup, output))
    current_doc += 1
    current_position_file.seek(0)
    current_position_file.write(str(current_doc))

print("COMPLETED!")