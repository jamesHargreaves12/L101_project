from tqdm import tqdm

from baseline.get_dataset import get_fever_dataset
from baseline.results import get_possible_docs

data = get_fever_dataset()
titles = []
for c, evidence in tqdm(data[:len(data) // 8]):
    possible_docs = get_possible_docs(c)
    titles.extend([x["title"] for x in possible_docs])

for t in titles:
    disamb = "(" in t and ")" in t
    if disamb:
        dis_text = t.replace("(", ")").split(")")[1]
        print(t, dis_text)

x = 1
