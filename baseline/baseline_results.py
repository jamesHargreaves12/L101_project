from time import time
import os
from tqdm import tqdm

from baseline.get_dataset import get_fever_dataset
from baseline.results import get_top_k_docs, get_bottom_k_docs, get_random_k_docs
from baseline.utils import get_json_str, extract_titles_from_evidence


def run_experiment(rank_method, k):
    file_list = filter(lambda x: x.startswith("baseline"), os.listdir("results"))
    latest_file = sorted(file_list, key=lambda x: float(x.split("_")[1].replace(".txt", "")))[-1]

    from_progress_file = False
    progress_file = latest_file

    dataset = get_fever_dataset(get_train=False, get_dev=True)
    file_path = progress_file if from_progress_file else "baseline_{}.txt".format(time())
    recall_file = open("results/" + file_path, "a+")
    recall_file.seek(0)
    progress_count = 0
    score_so_far = 0
    start_pos = 0
    for line in recall_file.readlines():
        progress_count += int(line.split(',')[3])
        score_so_far += int(line.split(',')[2])
        start_pos += 1
    total_recall = score_so_far
    total_docs = progress_count

    print("Starting From: ", start_pos)
    for claim, evidence in tqdm(dataset[start_pos:]):
        pred_titles = rank_method(claim, k)
        true_titles = extract_titles_from_evidence(evidence)
        pred_titles = list(map(lambda t: t['title'].replace(" ", "_"), pred_titles))
        count = 0
        for title in true_titles:
            if title in pred_titles:
                count += 1
        total_recall += count
        total_docs += len(true_titles)
        recall = count / len(true_titles) if true_titles else 1
        recall_file.write("{},{},{},{}\n".format(get_json_str(claim), recall, count, len(true_titles)))
        recall_file.flush()

    print(rank_method.__name__, k, ":")
    print(total_recall, total_docs, total_recall / total_docs)
    print()


# FOR K = 5
# get_top_k_docs()
#   9898 16016 = 0.618
# get_bottom_k_docs():
#   4800 16016 = 0.297
# get_random_k_docs():
#   8075 16016 = 0.504 - similar to drqa
for k in [1, 10]:
    run_experiment(get_top_k_docs, k)
    run_experiment(get_bottom_k_docs, k)
    run_experiment(get_random_k_docs, k)
# get_top_k_docs 1 :
# 9295 16016 0.5803571428571429
# get_bottom_k_docs 1 :
# 1148 16016 0.07167832167832168
# get_random_k_docs 1 :
# 3147 16016 0.196491008991009
#
# get_top_k_docs 10 :
# 10019 16016 0.625561938061938
# get_bottom_k_docs 10 :
# 7920 16016 0.4945054945054945
# get_random_k_docs 10 :
# 9549 16016 0.5962162837162838
