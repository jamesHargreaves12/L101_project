from baseline.get_dataset import get_fever_dataset



data = get_fever_dataset(get_train=False, get_dev=True)
count = 0
for c, evidence in data:
    min_ev = len(evidence[0])
    for e_1 in evidence:
        docs = set([x[2] for x in e_1])
        min_ev = min(min_ev, len(docs))
    if min_ev > 1:
        count += 1
print(count)