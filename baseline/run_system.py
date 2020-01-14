from random import random

from tqdm import tqdm
import os

from baseline.database import convert_to_form_in_db
from baseline.normalization import normalise_rows
from neon_test import MyModel

os.environ["CLASSPATH"] = "./DrQA/data/corenlp/*"

from baseline.get_dataset import get_fever_dataset
from baseline.mediawiki_api import get_pageview_data
from baseline.results import get_possible_docs, rank_data
from baseline.utils import get_overlap, extract_titles_from_evidence, get_mlp_features
from running_models.run_DrQA import get_drqa_score_sent, get_drqa_score_doc


def get_unique_docs(docs):
    title_to_doc = {}
    for x in docs:
        title_to_doc[x["title"]] = x
    return list(title_to_doc.values())


# open("data/nn_train_with_drqa.csv", "w"),open('data/tmp_train_recorder_with_drqa.txt', "w")


def get_top_titles_using_mlp(mlp_model, inputs, titles, k=5):
    if not inputs:
        # print("No Titles", titles)
        return []
    prob_being_true = [x[1] for x in mlp_model.get_results(normalise_rows(inputs))]
    pairs = zip(prob_being_true, titles)
    return [x[1] for x in sorted(pairs, reverse=True)[:k]]


def get_top_k_titles_mlp(docs, claim, mlp_model, k=5):
    all_titles = [x["title"] for x in docs]

    inputs = []
    for doc in get_unique_docs(docs):
        inputs.append(get_mlp_features(doc, claim, all_titles))

    return get_top_titles_using_mlp(mlp_model, inputs, all_titles, k=k)


def has_full_evidence(evidence, system_titles, k):
    full_evidence = False
    for e_1 in evidence:
        true_titles = [x[2] for x in e_1]
        if len(set(true_titles)) > k:
            continue
        full_evidence = full_evidence or all(map(lambda x: x in system_titles, true_titles))
    return full_evidence


if __name__ == "__main__":
    for with_caps_entities_check in [False]:
        for k in [5]:
            mlp_model = MyModel("model", batch_size=1)
            # if with_caps_entities_check:
            #     mlp_model.load_from_path("models/test/mlp-final_with_sent_ents.mdl")
            # else:
            #     mlp_model.load_from_path("models/test/mlp-final_no_sent_ents.mdl")
            mlp_model.load_from_path("models/test/mlp-final_no_sent_no_drqa.mdl")
            IS_TRAIN_DATASET = False
            dataset = get_fever_dataset(get_train=IS_TRAIN_DATASET, get_dev=not IS_TRAIN_DATASET)
            recall = 0
            baseline_beats = 0
            both_lose = 0
            oracle_loses = 0
            total = 0
            both_right = 0
            baseline_loses = 0
            full_evidence_mlp_count = 0
            full_evidence_baseline_count = 0
            full_evidence_oracle_count = 0
            total_tests = 0
            baseline_beats_claims = []
            av_oracle_len = 0
            oracle_recall = 0
            for claim, evidence in tqdm(dataset):
                total_tests += 1
                ev_titles = extract_titles_from_evidence(evidence)
                possible_docs = get_possible_docs(claim, with_caps_entities_check)

                top_titles = get_top_k_titles_mlp(possible_docs, claim, mlp_model, k=k)
                baseline_docs = rank_data(possible_docs, claim)[:k]
                baseline_titles = list(map(lambda t: t['title'].replace(" ", "_"), baseline_docs))
                all_titles = [x["title"] for x in possible_docs]
                av_oracle_len += len(set(all_titles))
                # if not top_titles:
                #     print("No Titles:",claim)
                top_titles = [convert_to_form_in_db(t) for t in top_titles]
                all_titles = [convert_to_form_in_db(t) for t in all_titles]
                if has_full_evidence(evidence, top_titles, k):
                    full_evidence_mlp_count += 1
                if has_full_evidence(evidence, baseline_titles, k):
                    full_evidence_baseline_count += 1
                if has_full_evidence(evidence, all_titles, k):
                    full_evidence_oracle_count += 1

                oracle_ev_count = 0

                for true_title in set(ev_titles):
                    total += 1
                    mlp_found = true_title in top_titles
                    baseline_found = true_title in baseline_titles
                    oracle_found = true_title in all_titles

                    if not oracle_found:
                        oracle_loses += 1
                    else:
                        oracle_ev_count += 1
                        if mlp_found:
                            recall += 1
                            if baseline_found:
                                both_right += 1
                            else:
                                baseline_loses += 1
                        else:
                            if baseline_found:
                                baseline_beats += 1
                            else:
                                both_lose += 1
                oracle_recall += min(oracle_ev_count, k)

            print(total, recall, recall / total)
            print("With cap ents", with_caps_entities_check)
            print("For k = ", k)
            print(baseline_beats, both_lose, oracle_loses, total, both_right, baseline_loses)
            print("Total documents = ", total)
            print("Oracle recall = ", oracle_recall, (oracle_recall)/total)
            print("Both right = ", both_right)
            print("Only MLP right = ", baseline_loses)
            print("Only Baseline right = ", baseline_beats)
            print("Both Wrong = ", both_lose)
            print("Recall MLP = ", (both_right+baseline_loses)/total)
            print("Recall baseline = ", (both_right+baseline_beats)/total)
            print("Full evidence counts = ", full_evidence_mlp_count, full_evidence_baseline_count, full_evidence_oracle_count)
            print("Avarage number docs from entity link = ", av_oracle_len)


# RUN SYSTEM FOR NO SENT ENTS
# With cap ents False
# For k =  1
# 1268 1395 3587 16016 8068 1698
# Total documents =  16016
# Oracle recall =  12429 0.7760364635364635
# Both right =  8068
# Only MLP right =  1698
# Only Baseline right =  1268
# Both Wrong =  1395
# Recall MLP =  0.6097652347652348
# Recall baseline =  0.5829170829170829
# Full evidence counts =  8969 8603 11040
# Avarage number docs from entity link =  168008
#
#
# With cap ents False
# For k =  5
# 113 181 3587 16016 9869 2266
# Total documents =  16016
# Oracle recall =  12429 0.7760364635364635
# Both right =  9869
# Only MLP right =  2266
# Only Baseline right =  113
# Both Wrong =  181
# Recall MLP =  0.7576798201798202
# Recall baseline =  0.6232517482517482
# Full evidence counts =  11060 9136 11271
# Avarage number docs from entity link =  168008
#
# For k =  10
# 43 69 3587 16016 10169 2148
# Total documents =  16016
# Oracle recall =  12429 0.7760364635364635
# Both right =  10169
# Only MLP right =  2148
# Only Baseline right =  43
# Both Wrong =  69
# Recall MLP =  0.7690434565434565
# Recall baseline =  0.6376123876123876
# Full evidence counts =  11187 9320 11271
# Avarage number docs from entity link =  168008



# With cap ents True

# With cap ents True
# For k =  1
# 1304 1588 3336 16016 7988 1800
# Total documents =  16016
# Oracle recall =  12680 0.7917082917082917
# Both right =  7988
# Only MLP right =  1800
# Only Baseline right =  1304
# Both Wrong =  1588
# Recall MLP =  0.6111388611388612
# Recall baseline =  0.5801698301698301
# Full evidence counts =  8985 8566 11233
# Avarage number docs from entity link =  224211

# With cap ents True
# For k =  5
# 169 284 3336 16016 9815 2412
# Total documents =  16016
# Oracle recall =  12680 0.7917082917082917
# Both right =  9815
# Only MLP right =  2412
# Only Baseline right =  169
# Both Wrong =  284
# Recall MLP =  0.7634240759240759
# Recall baseline =  0.6233766233766234
# Full evidence counts =  11138 9150 11487
# Avarage number docs from entity link =  224211
#
# With cap ents True
# For k =  10
# 80 133 3336 16016 10178 2289
# Total documents =  16016
# Oracle recall =  12680 0.7917082917082917
# Both right =  10178
# Only MLP right =  2289
# Only Baseline right =  80
# Both Wrong =  133
# Recall MLP =  0.7784090909090909
# Recall baseline =  0.6404845154845155
# Full evidence counts =  11327 9371 11487
# Avarage number docs from entity link =  224211
