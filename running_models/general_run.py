from random import random

from flask import json
from tqdm import tqdm

from baseline.get_dataset import get_fever_dataset
from baseline.run_system import has_full_evidence
from baseline.utils import extract_titles_from_evidence
from running_models.run_DrQA import get_titles, get_titles_sent


# Require CLASSPATH=./DrQA/data/corenlp/* in runtime

def get_drqa_results(k, first_sent_model=False):
    tests = get_fever_dataset(get_train=False, get_dev=True)
    num_recall = 0
    total_documents = 0
    fe = 0
    print("Starting Evaluation")
    fp_sig = open("data/sig_drqa_{}.txt".format(k), "w+")
    for claim, evidence in tqdm(tests):
        if first_sent_model:
            titles = get_titles_sent(claim, k)
        else:
            titles = get_titles(claim, k)

        evidence_titles = extract_titles_from_evidence(evidence)
        full_evidence = has_full_evidence(evidence, titles,k)
        # if full_evidence:
        #     fp_sig.write("1\n")
        # else:
        #     fp_sig.write("0\n")
        fe += 1 if full_evidence else 0
        num_recall += sum([1 for e in evidence_titles if e in titles])
        total_documents += len(evidence_titles)
        if random() < 0.001:
            print("Progress Check:", num_recall, total_documents, num_recall / total_documents)
    print(fe)
    return num_recall, total_documents, fe


if __name__ == "__main__":
    # Old:
    # On the training set:
    # DrQA = 68318 128148 0.5331179573618005 - this value is consistent with the paper
    # Redo:
    # On the training set
    # 68095 140085 0.4860977263804119
    # On dev set
    # 8354 16016 0.5216033966033966
    # Fairly certain that the papers results are on the test set not the dev set so happy with these results
    print("1 RESULTS:", get_drqa_results(1, True))
    print("5 RESULTS:", get_drqa_results(5, True))
    print("10 RESULTS:", get_drqa_results(10, True))

    # Final results(Dev set):
    # print("1 RESULTS:", get_drqa_results(1))
    # (4035, 16016) = 25.2%
    # print("5 RESULTS:", get_drqa_results(5))
    # 5 RESULTS: (8354, 16016) = 52.1%
    # print("10 RESULTS:", get_drqa_results(10))
    # 10 RESULTS: (9950, 16016) = 62.1%

    # print("Sent 1 RESULTS:", get_drqa_results(1, True))
    # Sent 1 RESULTS: (2198, 16016) = 13.7%
    # print("Sent 5 RESULTS:", get_drqa_results(5, True))
    # Sent 5 RESULTS: (4994, 16016) = 31.2%
    # print("Sent 10 RESULTS:", get_drqa_results(10, True))
    # Sent 10 RESULTS: (6313, 16016) = 39.4%

