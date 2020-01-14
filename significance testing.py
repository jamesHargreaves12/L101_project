import math
import random

from tqdm import tqdm


def next_exchange(cur_exchange):
    for i in range(len(cur_exchange)):
        if cur_exchange[i] == 0:
            cur_exchange[i] = 1
            return cur_exchange
        else:
            cur_exchange[i] = 0
    raise ValueError("No more values")


def get_diff_means(preds_a, preds_b, exchanges):
    tot_a = 0
    tot_b = 0
    for i,val in enumerate(exchanges):
        if val == 0:
            tot_a += preds_a[i]
            tot_b += preds_b[i]
        else:
            tot_b += preds_a[i]
            tot_a += preds_b[i]
    return abs(tot_a-tot_b)/len(preds_a)

def permutation_test(preds_a, preds_b):
    assert(len(preds_a) == len(preds_b))
    exchanges = [0 for _ in range(len(preds_a))]
    diff_means = get_diff_means(preds_a,preds_b,exchanges)
    more_extreme_count = 0
    try:
        while True:
            # print(exchanges)
            if get_diff_means(preds_a,preds_b,exchanges) >= diff_means:
                more_extreme_count += 1
            exchanges = next_exchange(exchanges)
    except:
        return more_extreme_count/ math.pow(2,len(preds_a))


def get_random_exchange(length):
    return [random.randint(0,1) for _ in range(length)]


def perm_test_random(preds_a, preds_b,num_perms):
    exchanges = [0 for _ in range(len(preds_a))]
    diff_means = get_diff_means(preds_a,preds_b,exchanges)
    more_extreme_count = 0
    for _ in tqdm(range(num_perms)):
        # print(exchanges)
        exchanges = get_random_exchange(len(preds_a))

        if get_diff_means(preds_a,preds_b,exchanges) >= diff_means:
            more_extreme_count += 1
    return more_extreme_count/num_perms


def get_results(path):
    results = []
    fp = open(path, "r")
    for line in fp.readlines():
        results.append(int(line))
    return results


baseline_5_res = get_results("data/sig_baseline_10.txt")
drqa_5_res = get_results("data/sig_drqa_1.txt")
print(sum(drqa_5_res))
# print(perm_test_random(baseline_5_res, drqa_5_res,10000))