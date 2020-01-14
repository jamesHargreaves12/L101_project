import random
import string

from spacy.lang.en import STOP_WORDS
from nltk import tokenize

from baseline import cache_builder
from baseline.allennlp_test import get_noun_phrases
from baseline.database import is_in_database
from baseline.get_dataset import get_fever_dataset
from baseline.mediawiki_api import get_doc_meta_for_query
from baseline.utils import get_overlap

FILTER_TITLES = True


def rank_data(data, claim, reversed=False):
    claim_words = set(claim.replace("_", " ").split(" "))
    return sorted(data, key=lambda x: get_overlap(x["title"], claim_words), reverse=not reversed)
    # initially percentage overlap on title
    # would be good to use some kind of nn for this eventualy


def filter_on_title(data, claim):
    # remove_punct:
    claim = claim.translate(str.maketrans('', '', string.punctuation))
    claim_words = set(claim.lower().replace("-", " ").split(" "))
    filtered = []
    for d in data:
        t = d["title"]
        words = set(t.lower().split(" "))
        words.discard("")
        non_stop = words.difference(STOP_WORDS)
        if non_stop and any(map(lambda x: x in claim_words, non_stop)) and is_in_database(t):
            filtered.append(d)
    return filtered


def get_possible_docs(claim, with_caps_ents=False):
    nps = get_noun_phrases(claim)
    nps.append(claim)
    if with_caps_ents:
        capitalised = get_capitalised_entities(claim)
        nps.extend(capitalised)
    data = []
    for ss in nps:
        np = filter(lambda word: not (word in STOP_WORDS or word == ""), ss.split(" "))
        if any(np):
            data.extend(get_doc_meta_for_query(ss))
    if FILTER_TITLES:
        return filter_on_title(data, claim)
    else:
        return data


def get_capitalised_entities(claim):
    words = tokenize.word_tokenize(claim)
    capitalised = map(lambda x: x[0].isupper(), words)
    entities = []
    current_capital = False
    start = 0
    for i, word in enumerate(capitalised):
        if word:
            if not current_capital:
                start = i
            current_capital = True
        else:
            if current_capital:
                entities.append(" ".join(words[start:i]))
            current_capital = False
    if current_capital:
        entities.append(" ".join(words[start:]))
    return entities


def get_top_k_docs(claim, k=5):
    data = get_possible_docs(claim)
    return rank_data(data, claim)[:k]


def get_bottom_k_docs(claim, k=5):
    data = get_possible_docs(claim)
    return rank_data(data, claim, reversed=True)[:k]


def get_random_k_docs(claim, k=5):
    data = get_possible_docs(claim)
    random.shuffle(data)
    return data[:k]


if __name__ == "__main__":
    # dataset = get_fever_dataset(limit=50)
    # for claim, evidence in dataset[:10]:
    #     titles = get_top_k_docs(claim)
    #     print(claim)
    #     print(titles)
    #     print()
    string = 'Ashley Graham was on a Television show in 2017 End'
    print(get_capitalised_entities(string))



