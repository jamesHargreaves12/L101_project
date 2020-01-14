from time import time
import os
from tqdm import tqdm

from baseline.get_dataset import get_fever_dataset
from baseline.results import get_top_k_docs, get_bottom_k_docs, get_random_k_docs
from baseline.run_system import has_full_evidence
from baseline.utils import get_json_str, extract_titles_from_evidence


def run_experiment(rank_method, k):
    dataset = get_fever_dataset(get_train=False, get_dev=True)
    total_recall = 0
    total_docs = 0
    total_full_evidence = 0
    print("Starting From: ", 0)
    fp_sig = open("data/sig_baseline_{}.txt".format(k), "w+")
    correct = []
    for claim, evidence in tqdm(dataset):
        pred_docs = rank_method(claim, k)
        true_titles = extract_titles_from_evidence(evidence)
        pred_titles = list(map(lambda t: t['title'].replace(" ", "_"), pred_docs))
        if has_full_evidence(evidence, pred_titles, k):
            total_full_evidence += 1
            correct.append(1)
            fp_sig.write("1\n")
        else:
            correct.append(0)
            fp_sig.write("0\n")
        count = 0
        for title in true_titles:
            if title in pred_titles:
                count += 1
        total_recall += count
        total_docs += len(true_titles)
    print(rank_method.__name__, k, ":")
    print(total_recall, total_docs, total_full_evidence, total_recall / total_docs, (total_full_evidence+6666)/19998)


for k in [1, 5, 10]:
    run_experiment(get_top_k_docs, k)
    # run_experiment(get_bottom_k_docs, k)
    # run_experiment(get_random_k_docs, k)
    # get_top_k_docs 1 :     9336 16016 8603 0.5829170829170829
    # get_top_k_docs 5 :     9982 16016 9136 0.6232517482517482
    # get_top_k_docs 10 :    10212 16016 9320 0.6376123876123876
    #
    # get_random_k_docs 1 :  1948   16016   0.12162837162837163
    # get_random_k_docs 5 :  6466   16016   0.4037212787212787
    # get_random_k_docs 10 : 8887   16016   0.5548826173826173
    #
    # get_bottom_k_docs 1 :  428    16016   0.026723276723276724
    # get_bottom_k_docs 5 :  2212   16016   0.1381118881118881
    # get_bottom_k_docs 10 : 5130   16016   0.3203046953046953
    #