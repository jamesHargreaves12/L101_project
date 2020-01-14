import csv
from random import random

from spacy.lang.en import STOP_WORDS
from nltk.stem import PorterStemmer

from baseline.mediawiki_api import get_pageview_data
from running_models.run_DrQA import get_drqa_score_doc, get_drqa_score_sent

ps = PorterStemmer()


def get_overlap(t, claim_words, ignore_stop=True):
    score = 0
    count = 0
    claim_words = list(map(ps.stem, claim_words))
    for word in map(ps.stem, t.split(" ")):
        if not ignore_stop or word not in STOP_WORDS:
            score += 1 if word in claim_words else 0
            count += 1
    if count == 0:
        return 0
    return score / count


def get_json_str(data):
    return str(data).replace(',', '_-comma-_').replace("\"", "_-dq-_").replace("'", '"')


def extract_titles_from_evidence(evidence):
    return set([x[2] for x in [sub_sub for sub in evidence for sub_sub in sub]])


def normalise(data):
    maximums = [0 for _ in data[0]]
    for d in data:
        for i, e in enumerate(d):
            maximums[i] = max(maximums[i], e)
    results = []
    for d in data:
        results.append([x / maximums[i] for i, x in enumerate(d)])
    return results


def get_normalised_data(file_path):
    x_train = []
    y_train = []
    with open(file_path, 'r') as fp:
        data = csv.reader(fp, delimiter=',')
        for row in data:
            if int(row[-1]) == 1 or random() < 39590 / 228522:
                row = list(map(lambda x: float(x), row))
                x_train.append(row[:-1])
                y_train.append([row[-1]])

    x_train = normalise(x_train)
    y_train = normalise(y_train)
    return x_train, y_train


def get_mlp_features(document, claim, other_titles_for_claim):
    claim_words = set(claim.replace("_", " ").split(" "))
    title = document["title"]
    document_title_id = title.replace(" ", "_")
    title_words = document_title_id.replace("_", " ").split(" ")

    occurrences = other_titles_for_claim.count(title)
    overlap = get_overlap(title, claim_words)
    page_views = get_pageview_data(document_title_id)
    drqa_doc = get_drqa_score_doc(claim, document_title_id)
    drqa_sent = get_drqa_score_sent(claim, document_title_id)

    has_disamb = 1 if "(" in title and ")" in title else 0
    disamb_str = title.replace("(", ")").split(")")[1] if has_disamb else ''
    disamb_overlap = get_overlap(disamb_str, claim_words) if has_disamb else 0
    # return (overlap, document["size_doc"], document["word_count_doc"], len(title_words), len(claim_words), occurrences,
    #         page_views, drqa_doc, drqa_sent, has_disamb, disamb_overlap)
    return (overlap, document["size_doc"], document["word_count_doc"], len(title_words), len(claim_words), occurrences,
            page_views, drqa_doc, has_disamb, disamb_overlap)
